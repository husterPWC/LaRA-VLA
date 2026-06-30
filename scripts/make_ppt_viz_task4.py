#!/usr/bin/env python
"""
PPT visualization: libero_10 task4 — Agent-view/Wrist RGB + Current/Future/Goal Mask
=====================================================================================
Generates multi-panel frames showing the spatial CoT training data at key moments
during a multi-step manipulation task.

Task: "put the white mug on the left plate and put the yellow and white mug on the right plate"
Subtasks:
  1. reach towards the white mug
  2. grasp the white mug
  3. reach towards the yellow and white mug
  4. put the yellow and white mug on the right plate

Output: output/ppt_viz_task4/  — PNG files ready for PPT insertion
"""

import json, os, sys
from pathlib import Path
import numpy as np
import imageio

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[1]
sys.path.insert(0, str(_REPO))

from lara_vla.data.spatial_cot_dataset import SpatialCoTDataset

SPATIAL = str(_REPO / "output" / "spatial_lara_libero")
INDEX = str(_REPO / "output" / "spatial_lara_libero_no_noops" / "spatial_lara_libero_index_cot_transition_all.jsonl")
COT = os.environ.get("LEROBOT_ROOT", str(_REPO.parent / "datasets" / "lovejuly" / "libero_lerobot_all"))
ALIGN = SPATIAL + "/cot_spatial_alignment.json"

OUT_DIR = _REPO / "output" / "ppt_viz_task4"

# Key frames: sample subtask boundaries + evenly spaced interior frames
KEY_FRAMES = [
    (0,   "00_start"),
    (10,  "01_reach_white"),
    (25,  "02_approach_white"),
    (42,  "03_grasp_white"),
    (55,  "04_hold_white"),
    (68,  "05_switch_yellow"),
    (90,  "06_reach_yellow"),
    (120, "07_grasp_yellow"),
    (150, "08_move_to_plate"),
    (180, "09_place_on_plate"),
    (200, "10_near_end"),
]


def make_overlay(rgb, mask, color=(0, 255, 0), alpha=0.45):
    """Draw colored mask overlay on RGB image."""
    rgb = rgb.astype(np.float32)
    mask_bool = mask.astype(bool)
    for c in range(3):
        rgb[mask_bool, c] = (1 - alpha) * rgb[mask_bool, c] + alpha * color[c]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def add_label(img, text, color=(255, 255, 255)):
    """Add a text label at the top of the image using simple pixel drawing."""
    h, w = img.shape[:2]
    # Draw a semi-transparent bar at the top
    bar_h = 26
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    bar[:] = (40, 40, 40)
    # Use numpy for simple text — we'll use imageio's pillow integration
    from PIL import Image, ImageDraw, ImageFont
    pil_img = Image.fromarray(np.vstack([bar, img]))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, 4), text, fill=color, font=font)
    return np.array(pil_img)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────
    print("Loading dataset...")
    ds = SpatialCoTDataset(SPATIAL, INDEX, COT, ALIGN, enable_dynamic_mask=True, cache_size=8)

    # Find entries for libero_10 task4 demo 19 (cot_ep 0)
    task4_entries = [
        (i, e) for i, e in enumerate(ds.entries)
        if e["suite"] == "libero_10" and e["task_id"] == 4
        and e["demo_id"] == 19
    ]
    task4_entries.sort(key=lambda x: x[1]["cot_frame_idx"])

    # Build frame index → dataset index map
    frame_to_dsidx = {e["cot_frame_idx"]: i for i, e in task4_entries}
    print(f"  Found {len(task4_entries)} frames for task4 demo19")

    # ── Episode meta ──────────────────────────────────────────
    meta_path = _REPO / "output" / "spatial_lara_libero" / task4_entries[0][1]["meta_path"]
    with open(meta_path) as f:
        meta = json.load(f)
    instance_to_id = meta.get("instance_to_id", {})
    objects_of_interest = meta.get("objects_of_interest", [])
    print(f"  Objects: {objects_of_interest}")

    # Load full NPZ for RGB access
    ep_path = _REPO / "output" / "spatial_lara_libero" / task4_entries[0][1]["episode_path"]
    ep_data = np.load(ep_path)
    print(f"  Episode frames: {ep_data['rgb_agentview'].shape[0]}")

    TASK_INSTRUCTION = "put the white mug on the left plate and put the yellow and white mug on the right plate"

    # ── Generate visualizations ────────────────────────────────
    from PIL import Image, ImageDraw, ImageFont
    try:
        font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_b = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font_s = ImageFont.load_default()
        font_b = ImageFont.load_default()
        font_title = font_b

    for cf, label in KEY_FRAMES:
        if cf not in frame_to_dsidx:
            print(f"  ⚠ Frame {cf} not found, skipping")
            continue
        ds_idx = frame_to_dsidx[cf]
        sample = ds[ds_idx]

        h5_idx = sample["hdf5_frame_idx"]
        fut_h5_idx = sample["hdf5_future_idx"]
        goal_idx = sample["subtask_end_idx"]

        # ── Get RGB at all three time points ────────────────────
        rgb_cur = ep_data["rgb_agentview"][h5_idx].copy()       # Current
        rgb_fut = ep_data["rgb_agentview"][fut_h5_idx].copy()   # Future (+8 frames)
        rgb_goal = (sample["goal_image_debug"].transpose(1, 2, 0) * 255).astype(np.uint8)  # 0-1→0-255
        rgb_wrist = ep_data["rgb_wrist"][h5_idx].copy()

        # Masks
        cur_mask_agent = sample["current_affordance_mask_agentview"].squeeze()
        cur_mask_wrist = sample["current_affordance_mask_wrist"].squeeze()
        fut_mask_agent = sample["future_affordance_mask_agentview"].squeeze()
        goal_mask_agent = sample["goal_affordance_mask_agentview"].squeeze()

        # ── Overlays: mask on ITS OWN RGB ───────────────────────
        cur_overlay = make_overlay(rgb_cur.copy(), cur_mask_agent, color=(0, 220, 0))    # green = current
        fut_overlay = make_overlay(rgb_fut.copy(), fut_mask_agent, color=(255, 200, 0))  # gold = future
        goal_overlay = make_overlay(rgb_goal.copy(), goal_mask_agent, color=(220, 80, 0))  # orange = goal

        # ── Build panel: 4 rows x 3 cols ────────────────────────
        H, W = 224, 224
        GAP = 4
        COLS = 3
        panel_w = W * COLS + GAP * (COLS - 1)
        INFO_H = 200  # Info bar height
        panel_h = H * 4 + GAP * 3 + INFO_H
        panel = np.ones((panel_h, panel_w, 3), dtype=np.uint8) * 30

        y = 0
        # Row 1: Agent RGB_current | RGB_future | RGB_goal
        panel[y:y+H, 0:W] = rgb_cur
        panel[y:y+H, W+GAP:2*W+GAP] = rgb_fut
        panel[y:y+H, 2*(W+GAP):3*W+2*GAP] = rgb_goal
        y += H + GAP

        # Row 2: Agent Mask Overlay
        panel[y:y+H, 0:W] = cur_overlay
        panel[y:y+H, W+GAP:2*W+GAP] = fut_overlay
        panel[y:y+H, 2*(W+GAP):3*W+2*GAP] = goal_overlay
        y += H + GAP

        # Row 3: Wrist RGB (current only — wrist cam doesn't shift much)
        #         + Wrist Cur Mask  +  Wrist Fut Mask
        wr_cur = make_overlay(rgb_wrist.copy(), cur_mask_wrist, color=(0, 220, 0))
        fut_wrist_mask = sample["future_affordance_mask_wrist"].squeeze()
        wr_fut = make_overlay(rgb_wrist.copy(), fut_wrist_mask, color=(255, 200, 0))
        goal_wrist_mask = sample["goal_affordance_mask_wrist"].squeeze()
        wr_goal = make_overlay(rgb_wrist.copy(), goal_wrist_mask, color=(220, 80, 0))
        panel[y:y+H, 0:W] = wr_cur
        panel[y:y+H, W+GAP:2*W+GAP] = wr_fut
        panel[y:y+H, 2*(W+GAP):3*W+2*GAP] = wr_goal
        y += H + GAP

        # Row 4: Wrist RGB x3 (all three time points same — wrist is fixed)
        panel[y:y+H, 0:W] = rgb_wrist
        panel[y:y+H, W+GAP:2*W+GAP] = rgb_wrist
        panel[y:y+H, 2*(W+GAP):3*W+2*GAP] = rgb_wrist
        y += H + GAP

        # ── Text annotations ────────────────────────────────────
        pil_panel = Image.fromarray(panel)
        draw = ImageDraw.Draw(pil_panel)

        # Column headers (inside each row, bottom)
        col_headers = [
            f"Current (h5_{h5_idx})",
            f"Future (h5_{fut_h5_idx}, +{fut_h5_idx-h5_idx})",
            f"Goal (h5_{goal_idx}, end)"
        ]
        for ri in range(4):
            row_y = ri * (H + GAP) + H - 16
            for ci, ch in enumerate(col_headers):
                draw.text((ci*(W+GAP)+4, row_y), ch, fill=(255,255,200), font=font_s)

        # Row labels on the right side
        row_labels = ["Agent RGB", "Agent Mask", "Wrist Mask", "Wrist RGB"]
        for ri, rl in enumerate(row_labels):
            draw.text((panel_w-110, ri*(H+GAP)+4), rl, fill=(180,180,180), font=font_s)

        # Info section at bottom
        y_info = 4*H + 3*GAP + 8
        cot_subtask = sample.get("cot_subtask", "")
        cot_text = sample.get("cot_text_transition", "")
        relation = sample.get("relation_label", "")

        draw.text((6, y_info), f"Task: {TASK_INSTRUCTION}", fill=(255,255,255), font=font_title)
        draw.text((6, y_info+22),
                  f"Frame {cf} | Subtask: \"{cot_subtask}\" | Relation: {relation} | Goal at frame {goal_idx}",
                  fill=(200,200,200), font=font_s)
        cot_display = cot_text[:300] + "..." if len(cot_text) > 300 else cot_text
        draw.text((6, y_info+40), f"CoT: {cot_display}", fill=(180,200,180), font=font_s)

        # Mask color legend
        draw.rectangle([6, y_info+62, 20, y_info+76], fill=(0, 200, 0), outline=(100,100,100))
        draw.text((24, y_info+60), "Current", fill=(0,220,0), font=font_s)
        draw.rectangle([90, y_info+62, 104, y_info+76], fill=(255, 180, 0), outline=(100,100,100))
        draw.text((108, y_info+60), "Future", fill=(255,200,0), font=font_s)
        draw.rectangle([174, y_info+62, 188, y_info+76], fill=(220, 70, 0), outline=(100,100,100))
        draw.text((192, y_info+60), "Goal", fill=(220,120,0), font=font_s)

        # Mask pixel counts
        n_cur = int(cur_mask_agent.sum())
        n_fut = int(fut_mask_agent.sum())
        n_goal = int(goal_mask_agent.sum())
        draw.text((6, y_info+82),
                  f"Mask px: cur={n_cur}  fut={n_fut}  goal={n_goal}  |  objects: {objects_of_interest}",
                  fill=(160,160,160), font=font_s)

        panel = np.array(pil_panel)
        out_path = OUT_DIR / f"task4_{label}.png"
        imageio.imwrite(str(out_path), panel)
        print(f"  ✅ {out_path}  | cur={n_cur}px  fut={n_fut}px  goal={n_goal}px")

    ep_data.close()

    # ── Summary ──────────────────────────────────────────────
    print(f"\nSaved to: {OUT_DIR}/")
    print(f"\n=== FOR PPT CAPTION ===")
    print(f"Task: {TASK_INSTRUCTION}")
    print(f"Subtasks:")
    print(f"  1. reach towards the white mug")
    print(f"  2. grasp the white mug")
    print(f"  3. reach towards the yellow and white mug")
    print(f"  4. put the yellow and white mug on the right plate")
    print(f"\nSpatial CoT example (frame 68):")
    sample68 = ds[frame_to_dsidx.get(68, list(frame_to_dsidx.values())[0])]
    print(f"  {sample68.get('cot_text_transition', 'N/A')}")


if __name__ == "__main__":
    main()
