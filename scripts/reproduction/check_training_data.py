"""验证 R3 使用的官方 LIBERO 训练数据链路。

该脚本直接调用仓库的 ``get_vla_dataset``、transform 和 reasoning formatter，
不实现替代 dataset/collator，也不构造假样本。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from laravla.dataloader.lerobot_datasets import get_vla_dataset


EXPECTED_DATASETS = {
    "libero_10_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_spatial_no_noops_1.0.0_lerobot",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_sample(dataset, stage: int) -> dict:
    trajectory_id, base_index = dataset.all_steps[0]
    raw = dataset.get_step_data(trajectory_id, base_index)
    transformed = dataset.transforms(raw)
    sample = dataset._build_sample_from_data(
        transformed, trajectory_id, base_index
    )

    action = np.asarray(sample["action"])
    if action.shape != (8, 7):
        raise RuntimeError(
            f"{dataset.dataset_name}: action shape 应为 (8, 7)，实际为 {action.shape}"
        )
    if not np.isfinite(action).all():
        raise RuntimeError(f"{dataset.dataset_name}: action 含 NaN/Inf")

    images = sample["image"]
    images_next = sample["image_next"]
    if len(images) != 2 or len(images_next) != 2:
        raise RuntimeError(
            f"{dataset.dataset_name}: 当前/下一帧相机数量应均为 2"
        )
    for kind, values in (("image", images), ("image_next", images_next)):
        for index, value in enumerate(values):
            array = np.asarray(value)
            if array.shape != (224, 224, 3) or array.dtype != np.uint8:
                raise RuntimeError(
                    f"{dataset.dataset_name}: {kind}[{index}] 为 "
                    f"{array.shape}/{array.dtype}"
                )
            if not np.isfinite(array).all() or float(array.std()) == 0.0:
                raise RuntimeError(
                    f"{dataset.dataset_name}: {kind}[{index}] 无有效视觉内容"
                )

    action_tokens = sample.get("action_tokens", "")
    if not action_tokens:
        raise RuntimeError(
            f"{dataset.dataset_name}: Stage {stage} 未生成 FAST action tokens"
        )
    if "<robot_action_" not in action_tokens:
        raise RuntimeError(
            f"{dataset.dataset_name}: FAST action token 格式异常"
        )
    if not sample.get("cot_available", False):
        raise RuntimeError(f"{dataset.dataset_name}: 抽取样本缺少 CoT 标注")
    if not sample.get("bbox_valid", False):
        raise RuntimeError(f"{dataset.dataset_name}: 抽取样本缺少有效 bbox 标注")

    language = sample["language"]
    expected_latent = max(stage - 1, 0)
    actual_latent = language.count("<|thinking|>")
    if actual_latent != expected_latent:
        raise RuntimeError(
            f"{dataset.dataset_name}: Stage {stage} 应有 {expected_latent} 个 "
            f"thinking token，实际为 {actual_latent}"
        )
    img_next_count = language.count("<img_next>")
    if img_next_count != 16:
        raise RuntimeError(
            f"{dataset.dataset_name}: img_next token 应为 16，实际为 {img_next_count}"
        )

    return {
        "dataset": dataset.dataset_name,
        "steps": len(dataset),
        "trajectories": len(dataset.trajectory_ids),
        "trajectory_id": int(trajectory_id),
        "base_index": int(base_index),
        "action_shape": list(action.shape),
        "action_dtype": str(action.dtype),
        "action_min": float(action.min()),
        "action_max": float(action.max()),
        "image_shapes": [list(np.asarray(image).shape) for image in images],
        "image_next_shapes": [
            list(np.asarray(image).shape) for image in images_next
        ],
        "image_next_fallback": bool(sample["image_next_fallback"]),
        "cot_available": bool(sample["cot_available"]),
        "bbox_valid": bool(sample["bbox_valid"]),
        "action_token_count": action_tokens.count("<robot_action_"),
        "thinking_token_count": actual_latent,
        "img_next_token_count": img_next_count,
        "formatted_instruction": language,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--fast-tokenizer", type=Path, required=True)
    parser.add_argument("--steps-cache-path", type=Path)
    parser.add_argument("--write-steps-cache", action="store_true")
    parser.add_argument("--stage", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve(strict=True)
    dataset_root = args.dataset_root.resolve(strict=True)
    fast_tokenizer = args.fast_tokenizer.resolve(strict=True)

    cfg = OmegaConf.load(config_path)
    cfg.datasets.vla_data.data_root_dir = str(dataset_root)
    cfg.datasets.vla_data.bridge_annotations.fast_tokenizer_name = str(
        fast_tokenizer
    )
    if args.steps_cache_path is not None:
        steps_cache_path = args.steps_cache_path.resolve()
        if args.write_steps_cache:
            steps_cache_path.mkdir(parents=True, exist_ok=True)
        cfg.datasets.vla_data.bridge_annotations.steps_cache_path = str(
            steps_cache_path
        )
    else:
        steps_cache_path = None
    cfg.datasets.vla_data.bridge_annotations.write_steps_cache = bool(
        args.write_steps_cache
    )
    cfg.datasets.vla_data.bridge_reasoning.stage = args.stage
    cfg.datasets.vla_data.bridge_reasoning.include_action_tokens = True

    mixture = get_vla_dataset(data_cfg=cfg.datasets.vla_data)
    names = {dataset.dataset_name for dataset in mixture.datasets}
    if names != EXPECTED_DATASETS:
        raise RuntimeError(
            f"libero_all 数据集合不一致：expected={sorted(EXPECTED_DATASETS)}, "
            f"actual={sorted(names)}"
        )

    samples = [validate_sample(dataset, args.stage) for dataset in mixture.datasets]
    report = {
        "status": "pass",
        "config": str(config_path),
        "dataset_root": str(dataset_root),
        "fast_tokenizer": {
            "path": str(fast_tokenizer),
            "revision": "ec4d7aa71691cac0b8bed6942be45684db2110f4",
            "tokenizer_json_sha256": sha256(fast_tokenizer / "tokenizer.json"),
        },
        "stage": args.stage,
        "steps_cache_path": (
            str(steps_cache_path) if steps_cache_path is not None else None
        ),
        "steps_cache_files": (
            [
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in sorted(steps_cache_path.glob("steps_*.pkl"))
            ]
            if steps_cache_path is not None
            else []
        ),
        "mixture_length": len(mixture),
        "dataset_sampling_weights": mixture.dataset_sampling_weights.tolist(),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
