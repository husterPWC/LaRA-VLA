#!/usr/bin/env python
"""
Generate filtered mask overlay videos for all demos in a suite.
Uses SpatialCoTDataset (gripper-based dynamic masking) for correct masks.

Output: output/mask_videos/{suite}/task_{tid}/demo_{did}_mask.mp4

Usage:
    python scripts/make_mask_videos.py --suite libero_10
    python scripts/make_mask_videos.py --suite all
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import imageio

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[1]
sys.path.insert(0, str(_REPO))

from lara_vla.data.spatial_cot_dataset import SpatialCoTDataset

SPATIAL = str(_REPO / "output" / "spatial_lara_libero")  # NPZ data root
INDEX = str(_REPO / "output" / "spatial_lara_libero_no_noops" / "spatial_lara_libero_index_cot_transition_all_fixed_v3.jsonl")
COT = os.environ.get('LEROBOT_ROOT',
                      str(_REPO.parent / 'datasets/lovejuly/libero_lerobot_all'))
ALIGN = SPATIAL + "/cot_spatial_alignment.json"
OUT_DIR = _REPO / "output" / "mask_videos"


def make_overlay(rgb, mask, color=(0, 255, 0), alpha=0.5):
    o = rgb.copy().astype(np.float32)
    m = mask.astype(bool)
    for c in range(3):
        o[m, c] = (1 - alpha) * o[m, c] + alpha * color[c]
    return np.clip(o, 0, 255).astype(np.uint8)


def build_video(ds, suite, task_id, demo_id):
    """Generate mask overlay video for one demo."""
    # Find entries for this demo
    indices = [i for i, e in enumerate(ds.entries)
               if e["suite"] == suite and e["task_id"] == task_id
               and e["demo_id"] == demo_id]
    indices.sort(key=lambda i: ds.entries[i].get("cot_frame_idx", 0))

    if not indices:
        return

    T = len(indices)
    out_dir = OUT_DIR / suite / f"task_{task_id:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / f"demo_{demo_id:06d}_mask.gif"

    # Collect all frames into memory
    frames = []
    for t, idx in enumerate(indices):
        s = ds[idx]
        h5f = s["hdf5_frame_idx"]
        rgb = ds._load_episode(ds.entries[idx]["episode_path"])["rgb_agentview"][h5f].copy()
        mask = s["current_affordance_mask_agentview"].squeeze()
        frames.append(make_overlay(rgb, mask))
        if t % 50 == 0 and t > 0:
            print(".", end="", flush=True)

    print(f" encoding {len(frames)} frames...", end="", flush=True)
    imageio.mimsave(str(gif_path), frames, fps=10, loop=0)
    size_mb = gif_path.stat().st_size / 1024 / 1024
    print(f" ✅ {size_mb:.1f}MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, default="libero_10",
                        help="Suite name or 'all'")
    parser.add_argument("--task-id", type=int, default=-1, help="-1 for all tasks")
    args = parser.parse_args()

    ds = SpatialCoTDataset(SPATIAL, INDEX, COT, ALIGN, enable_dynamic_mask=True)

    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"] \
        if args.suite == "all" else [args.suite]

    for suite in suites:
        # Get unique tasks and demos
        tasks = set()
        for e in ds.entries:
            if e["suite"] == suite:
                tasks.add((e["task_id"], e["demo_id"]))

        tasks = sorted(tasks)
        task_ids = sorted(set(t for t, d in tasks))
        if args.task_id >= 0:
            task_ids = [args.task_id]

        total = sum(1 for t, d in tasks if t in task_ids)
        count = 0
        for tid in task_ids:
            demos = sorted(d for t, d in tasks if t == tid)
            print(f"\n{suite}/task_{tid:02d}: {len(demos)} demos")
            for did in demos:
                count += 1
                print(f"  [{count}/{total}] demo_{did:06d}:", end="", flush=True)
                build_video(ds, suite, tid, did)

    print(f"\nDone. Videos saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
