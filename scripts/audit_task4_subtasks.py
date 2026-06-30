#!/usr/bin/env python
"""
Audit libero_10 task4: subtask boundary verification for ALL demos.
===================================================================
For each demo, prints the subtask timeline and saves key-frame visualizations
(Agent RGB + Current Mask + Goal RGB + Goal Mask) at every subtask boundary.

Output: output/audit_task4/  — one folder per demo

Usage:
    python scripts/audit_task4_subtasks.py
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

TASK_INSTRUCTION = "put the white mug on the left plate and put the yellow and white mug on the right plate"

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
    """Build a 2x2 comparison panel: RGB_cur | Mask_cur / RGB_goal | Mask_goal"""
    H, W = 224, 224
    panel = np.ones((H*2 + 120, W*2 + 4, 3), dtype=np.uint8) * 30

    cur_overlay = make_overlay(rgb_cur.copy(), cur_mask, color=(0, 220, 0))
    goal_overlay = make_overlay(rgb_goal.copy(), goal_mask, color=(220, 80, 0))

    panel[0:H, 0:W] = rgb_cur
    panel[0:H, W+4:2*W+4] = cur_overlay
    panel[H+4:2*H+4, 0:W] = rgb_goal
    panel[H+4:2*H+4, W+4:2*W+4] = goal_overlay

    # Labels
    pil_panel = Image.fromarray(panel)
    draw = ImageDraw.Draw(pil_panel)
    # Column headers
    draw.text((4, H-16), f"RGB cur (h5_{h5_frame})", fill=(255,255,200), font=FONT_S)
    draw.text((W+8, H-16), "Current Mask (green)", fill=(0,220,0), font=FONT_S)
    draw.text((4, 2*H-12), f"RGB goal (h5_{goal_frame})", fill=(255,255,200), font=FONT_S)
    draw.text((W+8, 2*H-12), "Goal Mask (orange)", fill=(220,120,0), font=FONT_S)

    # Info bar
    y = 2*H + 8
    draw.text((6, y), f"Frame {cot_frame} | Goal at {goal_frame} | Mask: cur={int(cur_mask.sum())}px goal={int(goal_mask.sum())}px",
              fill=(255,255,255), font=FONT_B)
    draw.text((6, y+22), f"{cot_text[:250]}", fill=(180,200,180), font=FONT_S)

    return np.array(pil_panel)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, default=INDEX_FIXED,
                        help=f"Index path (default: fixed v2)")
    parser.add_argument("--original", action="store_true",
                        help="Use original (unfixed) index")
    args = parser.parse_args()

    index_path = INDEX_DEFAULT if args.original else args.index
    out_root = _REPO / "output" / "audit_libero10_original" if args.original else OUT_ROOT
    print(f"Using index: {index_path}")
    print("Loading dataset...")
    ds = SpatialCoTDataset(SPATIAL, index_path, COT, ALIGN, enable_dynamic_mask=True, cache_size=8)

    # Get all demos for libero_10 task4
    demos = set()
    for e in ds.entries:
        if e["suite"] == "libero_10" and e["task_id"] == 4:
            demos.add(e["demo_id"])
    demos = sorted(demos)
    print(f"Task 4 has {len(demos)} demos: {demos}")

    # For each demo, collect the subtask timeline and boundary frames
    for did in demos:
        print(f"\n{'='*80}")
        print(f"=== DEMO {did} ===")
        print(f"{'='*80}")

        # Get all entries for this demo, sorted by frame
        entries = [(i, e) for i, e in enumerate(ds.entries)
                   if e["suite"] == "libero_10" and e["task_id"] == 4 and e["demo_id"] == did]
        entries.sort(key=lambda x: x[1]["cot_frame_idx"])

        if not entries:
            print("  No entries!")
            continue

        demo_dir = out_root / f"demo_{did:02d}"
        demo_dir.mkdir(parents=True, exist_ok=True)

        # Load NPZ for this demo
        ep_path = _REPO / "output" / "spatial_lara_libero" / entries[0][1]["episode_path"]
        ep_data = np.load(ep_path)
        T = ep_data["rgb_agentview"].shape[0]

        # Build subtask timeline
        prev_sub = None
        prev_goal = None
        timeline = []
        for idx, e in entries:
            cf = e["cot_frame_idx"]
            s = ds[idx]
            sub = s.get("cot_subtask", "")
            goal_idx = s["subtask_end_idx"]
            h5 = s["hdf5_frame_idx"]

            if sub != prev_sub or goal_idx != prev_goal:
                timeline.append({
                    "cot_frame": cf,
                    "h5_frame": h5,
                    "subtask": sub,
                    "goal_idx": goal_idx,
                    "cot_text": s.get("cot_text_transition", ""),
                    "relation": s.get("relation_label", ""),
                    "first_entry": True,
                })
            else:
                timeline.append({
                    "cot_frame": cf,
                    "h5_frame": h5,
                    "subtask": sub,
                    "goal_idx": goal_idx,
                    "cot_text": s.get("cot_text_transition", ""),
                    "relation": s.get("relation_label", ""),
                    "first_entry": False,
                })
            prev_sub = sub
            prev_goal = goal_idx

        # Print timeline (only boundary changes)
        print(f"  NPZ frames: {T}")
        print(f"  {'Frame':>5s} {'Cot':>4s} {'H5':>4s} {'Goal':>5s} {'Subtask'}")
        print(f"  {'-'*5} {'-'*4} {'-'*4} {'-'*5} {'-'*50}")
        boundary_indices = [j for j, t in enumerate(timeline) if t["first_entry"]]
        for j in boundary_indices:
            t = timeline[j]
            # Find end frame for this subtask segment
            end_cf = timeline[j+1]["cot_frame"] - 1 if j+1 < len(timeline) else timeline[-1]["cot_frame"]
            print(f"  {t['cot_frame']:4d}-{end_cf:<4d} {t['cot_frame']:4d} {t['h5_frame']:4d} {t['goal_idx']:5d} {t['subtask'][:50]}")

        # ── Save boundary frame visualizations ────────────────
        demo_dir.mkdir(parents=True, exist_ok=True)  # ensure exists
        for j in boundary_indices:
            t = timeline[j]
            cf = t["cot_frame"]
            h5 = t["h5_frame"]
            goal_idx = t["goal_idx"]
            sub = t["subtask"]

            ds_idx = None
            for i, e in entries:
                if e["cot_frame_idx"] == cf:
                    ds_idx = i
                    break
            if ds_idx is None:
                continue

            s = ds[ds_idx]
            rgb_cur = ep_data["rgb_agentview"][h5].copy()
            cur_mask = s["current_affordance_mask_agentview"].squeeze()
            goal_rgb = (s["goal_image_debug"].transpose(1, 2, 0) * 255).astype(np.uint8)
            goal_mask = s["goal_affordance_mask_agentview"].squeeze()
            cot_text = s.get("cot_text_transition", "")

            panel = build_boundary_panel(rgb_cur, cur_mask, goal_rgb, goal_mask,
                                          cf, h5, goal_idx, sub, cot_text)

            label = f"boundary_cf{cf:04d}_{sub[:30].replace(' ','_')}"
            out_path = demo_dir / f"{label}.png"
            imageio.imwrite(str(out_path), panel)
            print(f"  ✅ {out_path.name}")

        # ── Also save a mid-subtask frame for each segment ────
        for j in range(len(boundary_indices)):
            start_j = boundary_indices[j]
            end_j = boundary_indices[j+1] if j+1 < len(boundary_indices) else len(timeline)
            mid_j = (start_j + end_j) // 2
            if mid_j >= len(timeline):
                mid_j = len(timeline) - 1
            t = timeline[mid_j]

            cf = t["cot_frame"]
            h5 = t["h5_frame"]
            goal_idx = t["goal_idx"]
            sub = t["subtask"]

            ds_idx = None
            for i, e in entries:
                if e["cot_frame_idx"] == cf:
                    ds_idx = i
                    break
            if ds_idx is None:
                continue

            s = ds[ds_idx]
            rgb_cur = ep_data["rgb_agentview"][h5].copy()
            cur_mask = s["current_affordance_mask_agentview"].squeeze()
            goal_rgb = (s["goal_image_debug"].transpose(1, 2, 0) * 255).astype(np.uint8)
            goal_mask = s["goal_affordance_mask_agentview"].squeeze()
            cot_text = s.get("cot_text_transition", "")

            panel = build_boundary_panel(rgb_cur, cur_mask, goal_rgb, goal_mask,
                                          cf, h5, goal_idx, sub, cot_text)

            label = f"mid_cf{cf:04d}_{sub[:30].replace(' ','_')}"
            out_path = demo_dir / f"{label}.png"
            imageio.imwrite(str(out_path), panel)
            print(f"  ✅ {out_path.name}")

        # Save timeline as text
        with open(demo_dir / "timeline.txt", "w") as f:
            f.write(f"Task: {TASK_INSTRUCTION}\n")
            f.write(f"Demo: {did}\n")
            f.write(f"NPZ frames: {T}\n\n")
            f.write(f"{'Frame':>5s} {'H5':>4s} {'Goal':>5s} {'Subtask'}\n")
            f.write(f"{'-'*5} {'-'*4} {'-'*5} {'-'*50}\n")
            for j in boundary_indices:
                t = timeline[j]
                end_cf = timeline[j+1]["cot_frame"]-1 if j+1 < len(timeline) else timeline[-1]["cot_frame"]
                f.write(f"{t['cot_frame']:4d}-{end_cf:<4d} {t['h5_frame']:4d} {t['goal_idx']:5d} {t['subtask']}\n")
            f.write(f"\nAll frames details:\n")
            for t in timeline:
                f.write(f"  cf={t['cot_frame']:3d} h5={t['h5_frame']:3d} goal={t['goal_idx']:3d} "
                        f"rel={t['relation']:25s} sub={t['subtask'][:60]}\n")

        ep_data.close()

    print(f"\n{'='*80}")
    print(f"Done. Output: {out_root}/")
    print(f"Folders: {sorted([d.name for d in out_root.iterdir() if d.is_dir()])}")


if __name__ == "__main__":
    main()
