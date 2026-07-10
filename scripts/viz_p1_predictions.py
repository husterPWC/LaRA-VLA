#!/usr/bin/env python
"""
P1 Validation Visualization: save prediction samples for qualitative inspection.
================================================================================
Loads P1 best_model.pt, runs inference on val samples, saves comparison images.

Layout (3 rows):
  Row 1: RGB_cur | Current Mask
  Row 2: RGB_future + Pred Future Mask | RGB_future + GT Future Mask
  Row 3: RGB_goal + Pred Goal Mask     | RGB_goal + GT Goal Mask

All masks are overlaid on their own frame's RGB for direct visual comparison.

Usage:
    python scripts/viz_p1_predictions.py --num-samples 30 --output-dir results/P1_viz
"""

import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import imageio
from PIL import Image, ImageDraw, ImageFont

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[1]
sys.path.insert(0, str(_REPO))

import warnings; warnings.filterwarnings("ignore")

CKPT = os.environ.get("LARAVLA_CKPT",
    str(_REPO.parent / "models/LaRA-VLA-libero/checkpoints/steps_25000_pytorch_model.pt"))
SPATIAL = str(_REPO / "output" / "spatial_lara_libero")
IDX = str(_REPO / "output" / "spatial_lara_libero_no_noops" /
          "spatial_lara_libero_index_cot_transition_all_fixed_v3.jsonl")
COT = os.environ.get("LEROBOT_ROOT",
    str(_REPO.parent / "datasets" / "lovejuly" / "libero_lerobot_all"))

RELATION_NAMES = {0: "approach", 1: "grasp", 2: "release",
                  3: "place_inside", 4: "place_on_top",
                  5: "open_articulated", 6: "move_toward"}


def make_overlay(rgb, mask, color=(0, 255, 0), alpha=0.45):
    rgb = rgb.astype(np.float32)
    m = mask.astype(bool)
    for c in range(3):
        rgb[m, c] = (1 - alpha) * rgb[m, c] + alpha * color[c]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--p1-ckpt", type=str, default=str(_REPO / "results/P1_formal/best_model.pt"))
    parser.add_argument("--output-dir", type=str, default=str(_REPO / "results/P1_viz"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading P1 checkpoint: {args.p1_ckpt}")
    if not Path(args.p1_ckpt).exists():
        print(f"ERROR: P1 checkpoint not found: {args.p1_ckpt}")
        sys.exit(1)

    # ── Load VLA + inject P1 weights ─────────────────────────
    from laravla.model.tools import read_mode_config
    from omegaconf import OmegaConf
    model_cfg, _ = read_mode_config(Path(CKPT))
    model_cfg["framework"]["mask_conditioned_transition"] = {
        "enable": True, "num_mask_tokens": 8, "num_transition_tokens": 6,
        "mask_res": 56, "num_relation_labels": 6, "transition_dim": 512,
        "loss_weights": {"future_mask": 0.05, "goal_mask": 0.10, "relation": 0.05},
    }
    from laravla.model.framework import build_framework
    vla = build_framework(OmegaConf.create(model_cfg))
    vla.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)

    p1_state = torch.load(args.p1_ckpt, map_location="cpu")
    if "p1_state_dict" in p1_state:
        p1_state = p1_state["p1_state_dict"]
    vla.load_state_dict(p1_state, strict=False)
    vla = vla.to("cuda")
    vla.eval()
    for p in vla.parameters():
        p.requires_grad_(False)
    print("Model loaded with P1 weights.")

    # ── Load raw dataset for NPZ access ──────────────────────
    from lara_vla.data.spatial_lara_libero_dataset import SpatialLaRALiberoDataset
    base_ds = SpatialLaRALiberoDataset(SPATIAL, IDX)
    # Build episode cache
    ep_cache = {}
    def get_rgb(ep_path, h5_idx):
        if ep_path not in ep_cache:
            ep_cache[ep_path] = np.load(str(Path(SPATIAL) / ep_path))
        return ep_cache[ep_path]["rgb_agentview"][h5_idx].copy()

    # ── Build dataloader ─────────────────────────────────────
    from laravla.dataloader import build_dataloader
    loader = build_dataloader(OmegaConf.create({"datasets": {"vla_data": {
        "dataset_py": "spatial_cot_libero", "spatial_root": SPATIAL,
        "index_path": IDX, "cot_root": COT,
        "alignment_path": SPATIAL + "/cot_spatial_alignment.json",
        "enable_dynamic_mask": True, "per_device_batch_size": 1,
        "num_workers": 0, "state_dim": 7,
    }}}), dataset_py="spatial_cot_libero")

    try:
        FONT_S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        FONT_B = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except OSError:
        FONT_S = ImageFont.load_default()
        FONT_B = FONT_S

    saved = 0
    for batch in loader:
        if saved >= args.num_samples:
            break

        s = batch[0]
        rgb_cur_np = np.array(s["image"][0])

        cur_mask = torch.from_numpy(s["current_affordance_mask_agentview"]).unsqueeze(0).unsqueeze(0).to("cuda").float()
        future_gt_224 = s["future_affordance_mask_agentview"]
        goal_gt_224 = s["goal_affordance_mask_agentview"]
        rel_gt = s["relation_label_id"]
        suite = s["suite"]
        task_id = s["task_id"]
        demo_id = s["demo_id"]
        cur_h5 = s.get("hdf5_frame_idx", 0)
        fut_h5 = s.get("hdf5_future_idx", cur_h5 + 8)
        goal_h5 = s.get("subtask_end_idx", fut_h5)

        # ── Get RGB at future and goal frames ─────────────────
        # Find the entry for this sample
        ep_path = None
        for e in base_ds.entries:
            if (e["suite"] == suite and e["task_id"] == task_id
                and e["demo_id"] == demo_id and e["hdf5_frame_idx"] == cur_h5):
                ep_path = e["episode_path"]
                break
        if ep_path is None:
            continue

        rgb_fut_np = get_rgb(ep_path, fut_h5)
        rgb_goal_np = get_rgb(ep_path, goal_h5)

        # ── Frozen VLM → P1 forward ──────────────────────────
        with torch.no_grad():
            qwen_out = vla.qwen_vl_interface.encode_observation(
                images=[[s["image"][0]]], instructions=[s["lang"]],
                output_hidden_states=True,
            )
            vlm_hidden = qwen_out.hidden_states[-1]
            vlm_proj = vla.vlm_projector(vlm_hidden.float())
            mask_tokens = vla.mask_token_encoder(cur_mask)
            trans_tokens = vla.transition_module(vlm_proj, mask_tokens)
            future_logits = vla.future_mask_decoder(trans_tokens)
            goal_logits = vla.goal_mask_decoder(trans_tokens)
            rel_logits = vla.relation_head(trans_tokens)

        pred_future = (torch.sigmoid(future_logits) > 0.5).float().cpu().numpy()[0, 0]
        pred_goal = (torch.sigmoid(goal_logits) > 0.5).float().cpu().numpy()[0, 0]
        rel_pred_id = rel_logits.argmax(dim=1).item()

        # Upsample predictions to 224
        pred_future_224 = F.interpolate(
            torch.from_numpy(pred_future).unsqueeze(0).unsqueeze(0),
            size=(224, 224), mode='nearest').squeeze().numpy()
        pred_goal_224 = F.interpolate(
            torch.from_numpy(pred_goal).unsqueeze(0).unsqueeze(0),
            size=(224, 224), mode='nearest').squeeze().numpy()

        # ── Build panel ──────────────────────────────────────
        H, W = 224, 224
        panel = np.ones((H * 3 + 120, W * 2 + 4, 3), dtype=np.uint8) * 30

        # Row 1: RGB cur | Current Mask (on cur RGB)
        panel[0:H, 0:W] = rgb_cur_np
        panel[0:H, W+4:2*W+4] = make_overlay(rgb_cur_np.copy(), cur_mask[0,0].cpu().numpy(), (0, 220, 0))

        # Row 2: Pred Future (on future RGB) | GT Future (on future RGB)
        panel[H+4:2*H+4, 0:W] = make_overlay(rgb_fut_np.copy(), pred_future_224, (255, 200, 0))
        panel[H+4:2*H+4, W+4:2*W+4] = make_overlay(rgb_fut_np.copy(), future_gt_224, (255, 200, 0))

        # Row 3: Pred Goal (on goal RGB) | GT Goal (on goal RGB)
        panel[2*H+8:3*H+8, 0:W] = make_overlay(rgb_goal_np.copy(), pred_goal_224, (220, 80, 0))
        panel[2*H+8:3*H+8, W+4:2*W+4] = make_overlay(rgb_goal_np.copy(), goal_gt_224, (220, 80, 0))

        # ── Labels ───────────────────────────────────────────
        pil_panel = Image.fromarray(panel)
        draw = ImageDraw.Draw(pil_panel)
        draw.text((4, H-16), f"RGB cur (h5_{cur_h5})", fill=(255,255,200), font=FONT_S)
        draw.text((W+8, H-16), "Current Mask (green)", fill=(0,220,0), font=FONT_S)
        draw.text((4, 2*H-12), f"Pred Future on RGB fut (h5_{fut_h5})", fill=(255,200,0), font=FONT_S)
        draw.text((W+8, 2*H-12), f"GT Future on RGB fut", fill=(255,200,0), font=FONT_S)
        draw.text((4, 3*H+4-12), f"Pred Goal on RGB goal (h5_{goal_h5})", fill=(220,120,0), font=FONT_S)
        draw.text((W+8, 3*H+4-12), f"GT Goal on RGB goal", fill=(220,120,0), font=FONT_S)

        # Info bar
        y = 3 * H + 12
        rel_name = RELATION_NAMES.get(rel_pred_id, str(rel_pred_id))
        rel_gt_name = RELATION_NAMES.get(rel_gt, str(rel_gt))

        draw.text((6, y), f"{suite} task_{task_id} demo_{demo_id}  cur={cur_h5} fut={fut_h5} goal={goal_h5}",
                  fill=(255,255,255), font=FONT_B)
        draw.text((6, y+24), f"Pred Rel: {rel_name} | GT Rel: {rel_gt_name}",
                  fill=(180,200,180), font=FONT_S)
        draw.text((6, y+44),
                  f"Future px: pred={int(pred_future_224.sum())} gt={int(future_gt_224.sum())} | "
                  f"Goal px: pred={int(pred_goal_224.sum())} gt={int(goal_gt_224.sum())}",
                  fill=(160,160,160), font=FONT_S)

        out_path = output_dir / f"{suite}_t{task_id:02d}_d{demo_id:03d}_f{cur_h5:04d}.png"
        imageio.imwrite(str(out_path), np.array(pil_panel))
        saved += 1
        if saved % 10 == 0:
            print(f"  Saved {saved}/{args.num_samples}...")

    print(f"\nDone. {saved} samples saved to {output_dir}/")


if __name__ == "__main__":
    main()
