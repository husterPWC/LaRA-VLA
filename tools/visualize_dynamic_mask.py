#!/usr/bin/env python
"""
Visualize dynamic mask filtering for Spatial-CoLaRa dataset.
=============================================================
For a given demo, shows side-by-side comparison of:
  Left:  RGB + union mask (ALL objects of interest)
  Right: RGB + filtered mask (only objects relevant to CURRENT subtask)
  Caption: current subtask + relevant objects

Usage:
    python tools/visualize_dynamic_mask.py \
        --suite libero_10 --task-id 0 --demo-id 0 \
        --num-frames 8 --output-dir /tmp/dynamic_mask_viz

Author: Spatial-LaRA project
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import imageio

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[1]
sys.path.insert(0, str(_REPO))

from lara_vla.data.spatial_cot_dataset import SpatialCoTDataset, _objects_in_subtask
from lara_vla.data.spatial_lara_libero_dataset import SpatialLaRALiberoDataset

SPATIAL_ROOT = str(_REPO / "output" / "spatial_lara_libero")
INDEX_PATH = str(_REPO / "output" / "spatial_lara_libero" / "spatial_lara_libero_index_cot.jsonl")
COT_ROOT = "/home/robot/codePWC/lara_repro/datasets/lovejuly/libero_lerobot_all"
ALIGN_PATH = str(_REPO / "output" / "spatial_lara_libero" / "cot_spatial_alignment.json")


def make_overlay(rgb, mask, color=(0, 255, 0), alpha=0.5):
    """Draw mask overlay on RGB image."""
    rgb = rgb.copy().astype(np.float32)
    mask_bool = mask.astype(bool)
    for c in range(3):
        rgb[mask_bool, c] = (1 - alpha) * rgb[mask_bool, c] + alpha * color[c]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _resolve_seg_id(obj_name, instance_to_id):
    """Same logic as build_spatial_lara_libero.py"""
    if obj_name in instance_to_id:
        return instance_to_id[obj_name]
    for inst_name, sid in instance_to_id.items():
        if obj_name.startswith(inst_name):
            return sid
    for inst_name, sid in instance_to_id.items():
        if inst_name.startswith(obj_name):
            return sid
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--demo-id", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=8,
                        help="Number of frames to sample (evenly spaced)")
    parser.add_argument("--output-dir", type=str,
                        default=str(_REPO / "output" / "dynamic_mask_viz"))
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / f"{args.suite}_task{args.task_id}_demo{args.demo_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load base dataset (access raw NPZ, CoT-aligned index) ──
    base_ds = SpatialLaRALiberoDataset(SPATIAL_ROOT, INDEX_PATH)
    # Find entries for this demo (using cot_frame_idx from DTW-aligned index)
    indices = [
        i for i, e in enumerate(base_ds.entries)
        if e["suite"] == args.suite
        and e["task_id"] == args.task_id
        and e["demo_id"] == args.demo_id
    ]
    # Sort by cot_frame_idx
    indices.sort(key=lambda i: base_ds.entries[i].get("cot_frame_idx", 0))

    if not indices:
        print(f"No entries found for {args.suite}/task{args.task_id}/demo{args.demo_id}")
        return

    T = len(indices)
    print(f"Suite={args.suite} task={args.task_id} demo={args.demo_id} T={T}")

    # ── Load merged dataset with gripper-based dynamic filtering ──
    cot_ds = SpatialCoTDataset(SPATIAL_ROOT, INDEX_PATH, COT_ROOT, ALIGN_PATH,
                                enable_dynamic_mask=True)  # ← gripper-based filtering

    # ── Get instance_to_id from meta ───────────────────────────
    meta_path = (_REPO / "output" / "spatial_lara_libero"
                 / base_ds.entries[indices[0]]["meta_path"])
    with open(meta_path) as f:
        meta = json.load(f)
    instance_to_id = meta.get("instance_to_id", {})
    objects_of_interest = list(meta.get("objects_of_interest", []))

    # ── Load full episode NPZ for seg access ────────────────────
    ep_path = _REPO / "output" / "spatial_lara_libero" / base_ds.entries[indices[0]]["episode_path"]
    ep_data = np.load(ep_path)

    # ── Detect gripper state transitions (NOT subtask text!) ───
    gripper_states = []
    subtasks = []
    cot_frames = []
    for idx in indices:
        sample = cot_ds[idx]
        gripper_states.append(sample.get("cot_gripper_state", -1))
        subtasks.append(sample.get("cot_subtask", ""))
        cot_frames.append(sample.get("frame_idx", 0))

    # Find gripper change points + subtask boundaries
    boundaries = [0]
    for t in range(1, T):
        if gripper_states[t] != gripper_states[t - 1] or subtasks[t] != subtasks[t - 1]:
            boundaries.append(t)
    boundaries.append(T - 1)
    boundaries = sorted(set(boundaries))

    sample_frames = list(boundaries[:args.num_frames])
    if len(sample_frames) < args.num_frames:
        step = max(1, T // (args.num_frames - len(sample_frames)))
        extra = list(range(0, T, step))
        sample_frames = sorted(set(sample_frames + extra))[:args.num_frames]

    print(f"Boundaries: {len(boundaries)-1} segments, gripper changes at: {[b for b in boundaries if gripper_states[b] != gripper_states[max(0,b-1)]][:20]}")
    print(f"Selected frames: {sample_frames}")
    print()

    # ── Generate visualizations ─────────────────────────────────
    img_h, img_w = 224, 224

    for t in sample_frames:
        idx = indices[t]
        sample = cot_ds[idx]  # Uses gripper-based filtering from SpatialCoTDataset
        cot_frame_idx = sample["frame_idx"]  # CoT frame
        h5_frame_idx = sample["hdf5_frame_idx"]  # HDF5 frame (for NPZ access)

        # RGB from HDF5 frame (correct visual state)
        rgb_agentview = ep_data["rgb_agentview"][h5_frame_idx].copy()
        seg_agentview = ep_data["seg_agentview"][h5_frame_idx]

        # Union mask (all objects_of_interest)
        union_mask = np.zeros((img_h, img_w), dtype=bool)
        for obj_name in objects_of_interest:
            sid = _resolve_seg_id(obj_name, instance_to_id)
            if sid is not None:
                union_mask |= (seg_agentview == sid)

        # Filtered mask: use the already-computed mask from SpatialCoTDataset
        filtered_mask = sample["affordance_mask_agentview"].squeeze().astype(bool)

        # Build visualization
        union_overlay = make_overlay(rgb_agentview, union_mask, color=(255, 255, 0), alpha=0.4)   # yellow = union
        filter_overlay = make_overlay(rgb_agentview, filtered_mask, color=(0, 255, 0), alpha=0.5)  # green = filtered

        # Side-by-side: union (left) | filtered (right)
        h, w = rgb_agentview.shape[:2]
        combined = np.zeros((h + 40, w * 2, 3), dtype=np.uint8)
        combined[:h, :w] = union_overlay
        combined[:h, w:] = filter_overlay

        # Labels
        cot_subtask = sample.get("cot_subtask", "")
        n_rel = sample.get("num_relevant_objects", len(objects_of_interest))
        grip = sample.get("cot_gripper_state", -1)
        grip_label = {0: "CLOSED(holding)", 1: "OPEN(reaching)"}.get(grip, f"grip={grip}")

        # Save comparison image
        fname = f"cot_{cot_frame_idx:06d}_h5_{h5_frame_idx:06d}_mask_compare.png"
        imageio.imwrite(str(out_dir / fname), combined)

        # Save individual frames
        imageio.imwrite(str(out_dir / f"cot_{cot_frame_idx:06d}_h5_{h5_frame_idx:06d}_rgb.png"), rgb_agentview)
        imageio.imwrite(str(out_dir / f"cot_{cot_frame_idx:06d}_h5_{h5_frame_idx:06d}_union_mask.png"), union_overlay)
        imageio.imwrite(str(out_dir / f"cot_{cot_frame_idx:06d}_h5_{h5_frame_idx:06d}_filtered_mask.png"), filter_overlay)

        print(f"  CoT{cot_frame_idx:03d}→H5{h5_frame_idx:03d}: {grip_label}  "
              f"union={union_mask.sum()}px  filtered={filtered_mask.sum()}px  "
              f"n_rel={n_rel}  sub='{cot_subtask[:60]}'")

    ep_data.close()
    print(f"\nSaved {len(sample_frames)} visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
