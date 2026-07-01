#!/usr/bin/env python
"""
Audit all suites: subtask boundary verification for ALL tasks & demos.
======================================================================
For each suite/task/demo, prints the subtask timeline and saves
key-frame visualizations at every subtask boundary.

Output: output/audit_<suite>_fixed/<suite>_task_XX/demo_YY/

Usage:
    python scripts/audit_libero10_all_tasks.py                          # libero_10, all tasks
    python scripts/audit_libero10_all_tasks.py --task 4                 # libero_10 task 4 only
    python scripts/audit_libero10_all_tasks.py --suite libero_spatial   # libero_spatial all tasks
    python scripts/audit_libero10_all_tasks.py --suite libero_object
    python scripts/audit_libero10_all_tasks.py --suite libero_goal
"""

import argparse, json, os, sys
from pathlib import Path
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[1]
sys.path.insert(0, str(_REPO))

from lara_vla.data.spatial_cot_dataset import SpatialCoTDataset

SPATIAL = str(_REPO / "output" / "spatial_lara_libero")
INDEX_DEFAULT = str(_REPO / "output" / "spatial_lara_libero_no_noops" / "spatial_lara_libero_index_cot_transition_all.jsonl")
INDEX_FIXED = str(_REPO / "output" / "spatial_lara_libero_no_noops" / "spatial_lara_libero_index_cot_transition_all_fixed_v3.jsonl")
COT = os.environ.get("LEROBOT_ROOT", str(_REPO.parent / "datasets" / "lovejuly" / "libero_lerobot_all"))
ALIGN = SPATIAL + "/cot_spatial_alignment.json"
OUT_ROOT = _REPO / "output" / "audit_libero10_fixed"

try:
    FONT_S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    FONT_B = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
except OSError:
    FONT_S = ImageFont.load_default()
    FONT_B = FONT_S


def make_overlay(rgb, mask, color=(0, 255, 0), alpha=0.45):
    rgb = rgb.astype(np.float32)
    mask_bool = mask.astype(bool)
    for c in range(3):
        rgb[mask_bool, c] = (1 - alpha) * rgb[mask_bool, c] + alpha * color[c]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def build_boundary_panel(rgb_cur, cur_mask, rgb_goal, goal_mask,
                          cot_frame, h5_frame, goal_frame, subtask, cot_text):
    H, W = 224, 224
    panel = np.ones((H*2 + 130, W*2 + 4, 3), dtype=np.uint8) * 30

    cur_overlay = make_overlay(rgb_cur.copy(), cur_mask, color=(0, 220, 0))
    goal_overlay = make_overlay(rgb_goal.copy(), goal_mask, color=(220, 80, 0))

    panel[0:H, 0:W] = rgb_cur
    panel[0:H, W+4:2*W+4] = cur_overlay
    panel[H+4:2*H+4, 0:W] = rgb_goal
    panel[H+4:2*H+4, W+4:2*W+4] = goal_overlay

    pil_panel = Image.fromarray(panel)
    draw = ImageDraw.Draw(pil_panel)
    draw.text((4, H-16), f"RGB cur (h5_{h5_frame})", fill=(255,255,200), font=FONT_S)
    draw.text((W+8, H-16), "Current Mask (green)", fill=(0,220,0), font=FONT_S)
    draw.text((4, 2*H-12), f"RGB goal (h5_{goal_frame})", fill=(255,255,200), font=FONT_S)
    draw.text((W+8, 2*H-12), "Goal Mask (orange)", fill=(220,120,0), font=FONT_S)

    y = 2*H + 8
    draw.text((6, y), f"Frame {cot_frame} | Goal at {goal_frame} | Mask: cur={int(cur_mask.sum())}px goal={int(goal_mask.sum())}px",
              fill=(255,255,255), font=FONT_B)
    draw.text((6, y+22), f"{cot_text[:250]}", fill=(180,200,180), font=FONT_S)
    return np.array(pil_panel)


def audit_task(ds, suite, task_id, out_root):
    """Audit one task: all demos."""
    # Get task instruction from first entry
    task_entries_all = [e for e in ds.entries if e["suite"] == suite and e["task_id"] == task_id]
    if not task_entries_all:
        print(f"  No entries for task {task_id}!")
        return

    # Get instruction from CoT meta
    suite_key = (suite, task_entries_all[0]["cot_episode_id"])
    task_inst = ds._cot_tasks.get(suite_key, "N/A")
    print(f"\n{'#'*80}")
    print(f"# TASK {task_id}: {task_inst}")
    print(f"{'#'*80}")

    # Get all demos
    demos = sorted(set(e["demo_id"] for e in task_entries_all))
    print(f"  Demos: {len(demos)}")

    task_dir = out_root / f"{suite}_task_{task_id:02d}"
    task_dir.mkdir(parents=True, exist_ok=True)

    # Save task info
    with open(task_dir / "task_info.txt", "w") as f:
        f.write(f"Task {task_id}: {task_inst}\n")
        f.write(f"Demos: {demos}\n")

    for did in demos:
        entries = [(i, e) for i, e in enumerate(ds.entries)
                   if e["suite"] == suite and e["task_id"] == task_id and e["demo_id"] == did]
        entries.sort(key=lambda x: x[1]["cot_frame_idx"])

        if not entries:
            continue

        demo_dir = task_dir / f"demo_{did:02d}"
        demo_dir.mkdir(parents=True, exist_ok=True)

        # Load NPZ
        ep_path = _REPO / "output" / "spatial_lara_libero" / entries[0][1]["episode_path"]
        ep_data = np.load(ep_path)
        T = ep_data["rgb_agentview"].shape[0]

        # Build subtask timeline
        prev_sub = None; prev_goal = None
        timeline = []
        for idx, e in entries:
            s = ds[idx]
            cot_text = s.get("cot_text_transition", "")
            sub = cot_text.replace("Subtask: ", "").split("Reasoning:")[0].strip() if cot_text else ""
            goal_idx = s["subtask_end_idx"]
            h5 = s["hdf5_frame_idx"]
            timeline.append({
                "cf": e["cot_frame_idx"], "h5": h5,
                "subtask": sub, "goal_idx": goal_idx,
                "cot_text": s.get("cot_text_transition", ""),
                "relation": s.get("relation_label", ""),
                "new": (sub != prev_sub or goal_idx != prev_goal),
            })
            prev_sub = sub; prev_goal = goal_idx

        boundary_indices = [j for j, t in enumerate(timeline) if t["new"]]

        # Print timeline
        print(f"\n  --- Demo {did} (T={T}) ---")
        print(f"  {'Frame':>5s} {'Goal':>5s} {'Len':>4s} Subtask")
        print(f"  {'-'*5} {'-'*5} {'-'*4} {'-'*50}")
        for j in boundary_indices:
            t = timeline[j]
            end_cf = timeline[j+1]["cf"]-1 if j+1 < len(timeline) else timeline[-1]["cf"]
            seg_len = end_cf - t["cf"] + 1
            flag = " ⚠ SHORT!" if seg_len <= 3 else ""
            print(f"  {t['cf']:4d}-{end_cf:<4d} {t['goal_idx']:5d} {seg_len:4d} {t['subtask'][:50]}{flag}")

        # Save boundary images
        for j in boundary_indices:
            t = timeline[j]
            cf, h5, goal_idx, sub, cot_text = t["cf"], t["h5"], t["goal_idx"], t["subtask"], t["cot_text"]

            ds_idx = entries[0][0]  # fallback
            for i, e in entries:
                if e["cot_frame_idx"] == cf:
                    ds_idx = i; break

            s = ds[ds_idx]
            rgb_cur = ep_data["rgb_agentview"][h5].copy()
            cur_mask = s["current_affordance_mask_agentview"].squeeze()
            goal_rgb = (s["goal_image_debug"].transpose(1, 2, 0) * 255).astype(np.uint8)
            goal_mask = s["goal_affordance_mask_agentview"].squeeze()

            panel = build_boundary_panel(rgb_cur, cur_mask, goal_rgb, goal_mask,
                                          cf, h5, goal_idx, sub, cot_text)
            label = f"b_{cf:04d}_{sub[:25].replace(' ','_').replace('/','-')}"
            demo_dir.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(str(demo_dir / f"{label}.png"), panel)

        # Save timeline text
        with open(demo_dir / "timeline.txt", "w") as f:
            f.write(f"Task {task_id}: {task_inst}\nDemo {did}, T={T}\n\n")
            f.write(f"{'Frame':>5s} {'Goal':>5s} {'Len':>4s} Subtask\n")
            f.write(f"{'-'*5} {'-'*5} {'-'*4} {'-'*50}\n")
            for j in boundary_indices:
                t = timeline[j]
                end_cf = timeline[j+1]["cf"]-1 if j+1 < len(timeline) else timeline[-1]["cf"]
                seg_len = end_cf - t["cf"] + 1
                flag = " ⚠ SHORT!" if seg_len <= 3 else ""
                f.write(f"{t['cf']:4d}-{end_cf:<4d} {t['goal_idx']:5d} {seg_len:4d} {t['subtask']}{flag}\n")
            f.write(f"\nFrame-by-frame:\n")
            for t in timeline:
                f.write(f"  cf={t['cf']:3d} h5={t['h5']:3d} goal={t['goal_idx']:3d} rel={t['relation']:25s} sub={t['subtask'][:60]}\n")

        ep_data.close()
        print(f"  ✅ {len(boundary_indices)} boundary images saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, default="libero_10",
                        help="Suite: libero_10, libero_spatial, libero_object, libero_goal")
    parser.add_argument("--task", type=int, default=None, help="Single task ID, omit for all")
    parser.add_argument("--original", action="store_true", help="Use original (unfixed) index")
    args = parser.parse_args()

    suite = args.suite
    index_path = INDEX_DEFAULT if args.original else INDEX_FIXED
    out_root = OUT_ROOT if suite == "libero_10" else _REPO / "output" / f"audit_{suite}_fixed"
    print(f"Suite: {suite}")
    print(f"Using index: {index_path}")
    print("Loading dataset...")
    ds = SpatialCoTDataset(SPATIAL, index_path, COT, ALIGN, enable_dynamic_mask=True, cache_size=8)
    print("Done.\n")

    # Determine task range from data
    all_tids = sorted(set(e["task_id"] for e in ds.entries if e["suite"] == suite))
    tasks = [args.task] if args.task is not None else all_tids
    print(f"Tasks: {tasks}")
    for tid in tasks:
        audit_task(ds, suite, tid, out_root)

    print(f"\n{'='*80}")
    print(f"All done. Output: {out_root}/")


if __name__ == "__main__":
    main()
