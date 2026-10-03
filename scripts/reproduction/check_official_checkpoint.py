"""Load and inspect the released LaRA checkpoint before an R1 rollout.

This uses the official ``baseframework.from_pretrained`` path.  It does not
substitute model components or run a synthetic policy forward.
"""

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch

from laravla.dataloader.gr00t_lerobot.video import get_frames_by_timestamps
from laravla.model.framework.base_framework import baseframework
from examples.LIBERO.model2libero_interface import M1Inference


TOKENS = (
    "<|thinking|>",
    "<|start_of_thinking|>",
    "<|end_of_thinking|>",
    "<img_next>",
)


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def _version(package: str) -> str:
    return metadata.version(package)


def _first_video(dataset_root: Path) -> Path:
    suite = dataset_root / "libero_goal_no_noops_1.0.0_lerobot" / "videos"
    try:
        return next(suite.rglob("*.mp4"))
    except StopIteration as exc:
        raise FileNotFoundError(f"No LIBERO Goal video under {suite}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve(strict=True)
    dataset_root = args.dataset_root.resolve(strict=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = baseframework.from_pretrained(str(checkpoint))
    load_seconds = time.perf_counter() - started

    # A second CPU read makes the strictness result machine-readable.  The
    # official loader has already performed its own strict=True load above.
    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model_keys = set(model.state_dict())
    checkpoint_keys = set(checkpoint_state)
    missing_keys = sorted(model_keys - checkpoint_keys)
    unexpected_keys = sorted(checkpoint_keys - model_keys)
    del checkpoint_state
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Checkpoint is not an exact state_dict match: "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )

    model = model.to(torch.bfloat16).to("cuda").eval()
    torch.cuda.synchronize()
    interface = model.qwen_vl_interface
    tokenizer = interface.processor.tokenizer
    token_ids = {token: tokenizer.convert_tokens_to_ids(token) for token in TOKENS}
    for token, token_id in token_ids.items():
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise RuntimeError(f"Tokenizer did not resolve {token}: {token_id}")

    video = _first_video(dataset_root)
    frame = get_frames_by_timestamps(
        str(video), np.array([0.0]), video_backend="torchvision_av"
    )[0]
    if frame.shape != (256, 256, 3) or frame.dtype != np.uint8:
        raise RuntimeError(f"Unexpected decoded frame: {frame.shape} {frame.dtype}")
    client_format = M1Inference(
        policy_ckpt_path=str(checkpoint),
        enable_latent_reasoning=True,
        cot_mode="implicit",
        connect_server=False,
    )
    formatted_instruction = client_format.format_instruction(
        "put the bowl on the plate"
    )
    inputs = interface.build_qwenvl_inputs(
        images=[[Image.fromarray(frame)]],
        instructions=[formatted_instruction],
    )
    input_ids = inputs["input_ids"]
    token_counts = {
        token: int((input_ids == token_id).sum().item())
        for token, token_id in token_ids.items()
    }
    expected_counts = {
        "<|thinking|>": 3,
        "<|start_of_thinking|>": 1,
        "<|end_of_thinking|>": 1,
        "<img_next>": 16,
    }
    if token_counts != expected_counts:
        raise RuntimeError(
            f"Latent inference prompt mismatch: expected {expected_counts}, got {token_counts}"
        )

    normalization_key = next(iter(model.norm_stats))
    action_stats = model.norm_stats[normalization_key]["action"]
    action_cfg = model.config.framework.action_model
    params = list(model.parameters())
    report = {
        "status": "pass",
        "git_commit": _git_sha(),
        "checkpoint": str(checkpoint),
        "strict_state_dict_match": True,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "load_seconds": load_seconds,
        "versions": {
            "python_torch": torch.__version__,
            "transformers": _version("transformers"),
        },
        "gpu": {
            "name": torch.cuda.get_device_name(),
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "parameters": {
            "total": sum(p.numel() for p in params),
            "trainable": sum(p.numel() for p in params if p.requires_grad),
        },
        "model": {
            "class": type(model).__name__,
            "backbone_class": type(interface.model).__name__,
            "processor_class": type(interface.processor).__name__,
            "latent_reasoning_enabled": bool(
                model.config.framework.enable_latent_reasoning
            ),
            "image_next_enabled": bool(model.config.framework.img_next.enable),
            "image_next_teacher": bool(model.config.framework.img_next.use_teacher),
            "action_dim": int(action_cfg.action_dim),
            "state_dim": int(action_cfg.state_dim),
            "action_horizon": int(action_cfg.action_horizon),
            "future_action_window_size": int(action_cfg.future_action_window_size),
        },
        "tokenizer": {
            "token_ids": token_ids,
            "inference_prompt_token_counts": token_counts,
            "formatted_instruction": formatted_instruction,
            "input_ids_shape": list(input_ids.shape),
            "pixel_values_shape": list(inputs["pixel_values"].shape),
            "image_grid_thw": inputs["image_grid_thw"].tolist(),
        },
        "real_frame": {
            "source_video": str(video),
            "shape": list(frame.shape),
            "dtype": str(frame.dtype),
            "standard_deviation": float(frame.std()),
        },
        "normalization": {
            "key": normalization_key,
            "q01": action_stats["q01"],
            "q99": action_stats["q99"],
            "mask": action_stats["mask"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
