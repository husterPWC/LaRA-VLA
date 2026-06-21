#!/usr/bin/env python
"""
Spatial-LaRA Dataset Checker
============================
Validate all episode_*.npz files in a dataset directory.
Checks: loading, shapes, consistency, mask visibility, pose validity, future indices.

Usage:
    python tools/check_spatial_lara_dataset.py \
        --root output/spatial_lara_libero/libero_spatial/task_00

    python tools/check_spatial_lara_dataset.py \
        --root output/spatial_lara_libero/libero_spatial
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def check_episode(npz_path, meta_path, verbose=False):
    """Check a single episode. Returns dict of stats or None if failed."""
    try:
        d = np.load(npz_path, allow_pickle=False)
    except Exception as e:
        print(f"  ❌ Failed to load {npz_path.name}: {e}")
        return None

    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        meta = {}

    errors = []
    T = None

    # 1. Check all arrays exist and have consistent T
    expected_keys = [
        "rgb_agentview", "rgb_wrist",
        "seg_agentview", "seg_wrist",
        "affordance_mask_agentview", "affordance_mask_wrist",
        "primary_pose_world", "primary_pose_eef",
        "interest_poses_world", "interest_poses_eef",
        "robot0_eef_pos", "robot0_eef_quat",
        "robot0_gripper_qpos", "robot0_joint_pos",
        "actions", "future_indices",
    ]
    for key in expected_keys:
        if key not in d.files:
            errors.append(f"missing key: {key}")
            continue
        arr = d[key]
        if T is None:
            T = arr.shape[0]
        elif arr.shape[0] != T:
            errors.append(f"{key}: T={arr.shape[0]} != expected {T}")

    if T is None:
        return None

    # 2. RGB shape check
    for k in ["rgb_agentview", "rgb_wrist"]:
        if k in d.files and d[k].shape[-1] != 3:
            errors.append(f"{k}: last dim != 3")

    # 3. Affordance mask visibility
    mask_a = d["affordance_mask_agentview"]
    mask_w = d["affordance_mask_wrist"]
    agentview_visible = (mask_a.sum(axis=(1, 2)) > 0).sum()
    wrist_visible = (mask_w.sum(axis=(1, 2)) > 0).sum()
    agentview_rate = agentview_visible / T
    wrist_rate = wrist_visible / T

    if agentview_rate < 0.5:
        errors.append(f"agentview mask visible in only {agentview_visible}/{T} frames ({agentview_rate:.1%})")

    # 4. Pose validity
    pose = d["primary_pose_world"]
    pose_nan = np.isnan(pose).any(axis=1).sum()
    if pose_nan > 0:
        errors.append(f"pose NaN in {pose_nan}/{T} frames")

    # 5. Future indices
    fut = d["future_indices"]
    fut_invalid = ((fut < 0) | (fut >= T)).sum()
    if fut_invalid > 0:
        errors.append(f"future_indices invalid in {fut_invalid}/{T} frames")

    # 6. Actions
    actions = d["actions"]
    action_nan = np.isnan(actions).any()
    if action_nan:
        errors.append("actions contain NaN")

    # 7. Meta primary_object check
    primary = meta.get("primary_object", "")
    if not primary:
        errors.append("meta: empty primary_object")

    if errors and verbose:
        for e in errors:
            print(f"    ⚠ {e}")

    d.close()
    return {
        "T": T,
        "agentview_visible": agentview_visible,
        "agentview_rate": agentview_rate,
        "wrist_visible": wrist_visible,
        "wrist_rate": wrist_rate,
        "pose_nan": pose_nan,
        "fut_invalid": fut_invalid,
        "action_nan": action_nan,
        "has_primary": bool(primary),
        "num_errors": len(errors),
        "errors": errors if verbose else [],
    }


def check_task(task_dir, verbose=False):
    """Check all episodes in a task directory."""
    npz_files = sorted(task_dir.glob("demo_*/episode_*.npz"))
    if not npz_files:
        print(f"  No episodes found in {task_dir}")
        return None

    stats_list = []
    for npz_path in npz_files:
        meta_path = npz_path.parent / npz_path.name.replace(".npz", "_meta.json")
        stats = check_episode(npz_path, meta_path, verbose=verbose)
        if stats:
            stats_list.append(stats)

    if not stats_list:
        return None

    T_sum = sum(s["T"] for s in stats_list)
    agentview_ok = sum(1 for s in stats_list if s["agentview_rate"] >= 0.5)
    total_errors = sum(s["num_errors"] for s in stats_list)

    summary = {
        "num_demos": len(stats_list),
        "total_frames": T_sum,
        "avg_episode_len": T_sum / len(stats_list),
        "agentview_visible_demos": agentview_ok,
        "agentview_visible_rate": sum(s["agentview_rate"] for s in stats_list) / len(stats_list),
        "wrist_visible_rate": sum(s["wrist_rate"] for s in stats_list) / len(stats_list),
        "pose_nan_count": sum(s["pose_nan"] for s in stats_list),
        "future_invalid_count": sum(s["fut_invalid"] for s in stats_list),
        "action_nan_any": any(s["action_nan"] for s in stats_list),
        "total_errors": total_errors,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Spatial-LaRA Dataset Checker")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"❌ Root not found: {root}")
        sys.exit(1)

    # Detect: single task, suite, or full dataset
    task_dirs = sorted(root.glob("task_*"))

    if not task_dirs:
        # Root itself might be a task directory
        task_dirs = [root] if list(root.glob("demo_*/episode_*.npz")) else []

    if not task_dirs:
        # Check subdirectories (suite/task_xx structure)
        suite_dirs = [d for d in root.iterdir() if d.is_dir()]
        for sd in suite_dirs:
            task_dirs.extend(sorted(sd.glob("task_*")))

    if not task_dirs:
        print(f"❌ No task_* directories found under {root}")
        sys.exit(1)

    print(f"Checking {len(task_dirs)} task(s)...")
    print()

    overall_stats = []
    for task_dir in task_dirs:
        task_name = task_dir.name if task_dir.name.startswith("task_") else task_dir.parent.name + "/" + task_dir.name
        task_name = str(task_dir.relative_to(root)) if task_dir != root else task_dir.name
        print(f"  {task_name}: ", end="", flush=True)
        summary = check_task(task_dir, verbose=args.verbose)
        if summary is None:
            print("❌ NO VALID EPISODES")
            continue
        overall_stats.append(summary)

        status = "✅" if summary["total_errors"] == 0 else f"⚠ {summary['total_errors']} errors"
        print(f"{status}  "
              f"demos={summary['num_demos']}  "
              f"frames={summary['total_frames']}  "
              f"avg_len={summary['avg_episode_len']:.1f}  "
              f"ag_vis={summary['agentview_visible_rate']:.1%}  "
              f"wr_vis={summary['wrist_visible_rate']:.1%}  "
              f"pose_nan={summary['pose_nan_count']}  "
              f"fut_inv={summary['future_invalid_count']}")

    if not overall_stats:
        print("\n❌ No valid episodes found!")
        sys.exit(1)

    # Overall summary
    print()
    print("=" * 72)
    print("OVERALL SUMMARY")
    print("=" * 72)
    total_demos = sum(s["num_demos"] for s in overall_stats)
    total_frames = sum(s["total_frames"] for s in overall_stats)
    total_errors = sum(s["total_errors"] for s in overall_stats)
    print(f"  num_tasks:          {len(overall_stats)}")
    print(f"  num_demos:          {total_demos}")
    print(f"  total_frames:       {total_frames}")
    print(f"  avg_episode_len:    {total_frames / total_demos:.1f}")
    print(f"  agentview_visible:  {sum(s['agentview_visible_rate'] for s in overall_stats) / len(overall_stats):.1%}")
    print(f"  wrist_visible:      {sum(s['wrist_visible_rate'] for s in overall_stats) / len(overall_stats):.1%}")
    print(f"  pose_nan_count:     {sum(s['pose_nan_count'] for s in overall_stats)}")
    print(f"  future_invalid:     {sum(s['future_invalid_count'] for s in overall_stats)}")
    print(f"  action_nan_any:     {any(s['action_nan_any'] for s in overall_stats)}")
    print(f"  total_errors:       {total_errors}")

    if total_errors == 0:
        print(f"\n✅ All checks passed!")
    else:
        print(f"\n⚠ {total_errors} total errors found. Re-run with --verbose for details.")


if __name__ == "__main__":
    main()
