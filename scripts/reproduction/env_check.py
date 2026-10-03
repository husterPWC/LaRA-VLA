"""R0 checks only: official imports, CUDA arithmetic, and LIBERO reset/render.

No model substitute, training, policy prediction, or evaluation rollout.
"""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys


def versions(names):
    print(json.dumps({name: metadata.version(name) for name in names}, indent=2))
    print("python", sys.version, "executable", sys.executable)


def server():
    import numpy as np
    import torch
    import torchvision
    from pytorch3d import _C
    from pytorch3d.ops import knn_points
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from laravla.training.train import main
    from laravla.model.framework.laravla import Qwen_GR00T
    from laravla.model.modules.vlm.QWen3 import _QWen3_VL_Interface
    from laravla.dataloader.lerobot_datasets import get_vla_dataset
    from laravla.dataloader.gr00t_lerobot.video import get_frames_by_timestamps
    from deployment.model_server.server_policy import build_argparser

    versions(["torch", "torchvision", "transformers", "accelerate", "deepspeed",
              "numpy", "tokenizers", "huggingface_hub", "pyarrow", "pytorch3d"])
    assert torch.cuda.is_available(), "CUDA unavailable"
    x = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
    y = x @ x
    torch.cuda.synchronize()
    assert torch.isfinite(y).all() and (y == 32).all()
    # Native dependency ABI/CUDA probe, not a policy or model forward.
    points = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]], device="cuda")
    neighbors = knn_points(points, points, K=1)
    assert torch.equal(neighbors.idx, torch.tensor([[[0], [1]]], device="cuda"))
    assert (neighbors.dists == 0).all()
    print("PyTorch3D native CUDA extension", _C.__file__)
    print("GPU", torch.cuda.get_device_name(), "runtime", torch.version.cuda,
          "cuDNN", torch.backends.cudnn.version())
    print("model source", sys.modules[Qwen_GR00T.__module__].__file__)
    model_path = os.environ.get("LARA_BACKBONE")
    if model_path:
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        for token in ("<|thinking|>", "<|start_of_thinking|>", "<|end_of_thinking|>", "<img_next>"):
            print("backbone token", token, processor.tokenizer.convert_tokens_to_ids(token))
        print("processor", type(processor).__name__)
    dataset_root = Path(os.environ["LARA_DATASET_ROOT"])
    video = next((dataset_root / "libero_spatial_no_noops_1.0.0_lerobot" / "videos").rglob("*.mp4"))
    frames = get_frames_by_timestamps(
        str(video), np.array([0.0, 1.0, 2.0]), video_backend="torchvision_av"
    )
    assert frames.shape == (3, 256, 256, 3) and frames.dtype == np.uint8
    assert np.isfinite(frames).all() and frames.std() > 0
    print("torchvision_av AV1", frames.shape, str(frames.dtype), "std", float(frames.std()))
    print("R0 SERVER PASS (imports and CUDA only; checkpoint not loaded)")


def client():
    import numpy as np
    import mujoco
    from libero.libero import benchmark, get_libero_path
    from examples.LIBERO.eval_libero import _get_libero_env

    versions(["torch", "torchvision", "numpy", "mujoco", "robosuite", "libero"])
    print("LIBERO source", benchmark.__file__)
    suites = benchmark.get_benchmark_dict()
    for name in ("libero_spatial", "libero_goal", "libero_object", "libero_10"):
        suite = suites[name]()
        assert suite.n_tasks == 10
        for task_id in range(suite.n_tasks):
            states = suite.get_task_init_states(task_id)
            assert len(states) >= 50
        print(name, "10 tasks; all initial states load")
    suite = suites["libero_goal"]()
    env, description = _get_libero_env(suite.get_task(0), 256, 7)
    try:
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(0)[0])
        for name in ("agentview_image", "robot0_eye_in_hand_image"):
            frame = obs[name]
            assert frame.shape == (256, 256, 3) and frame.dtype == np.uint8
            assert np.isfinite(frame).all() and frame.std() > 0
            print(name, frame.shape, str(frame.dtype), "std", float(frame.std()))
        print("task", description, "MUJOCO_GL", os.environ.get("MUJOCO_GL"))
    finally:
        env.close()
    print("R0 CLIENT PASS (initial states + EGL reset/render; no rollout)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("component", choices=["server", "client"])
    args = parser.parse_args()
    {"server": server, "client": client}[args.component]()
