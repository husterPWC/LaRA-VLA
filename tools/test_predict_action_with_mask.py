#!/usr/bin/env python
"""
P3 Inference Smoke: predict_action with current_mask.
======================================================
Loads P2 best model, tests predict_action with GT current_mask input.
Verifies output shape, and compares with/without mask.

Usage:
    python tools/test_predict_action_with_mask.py --p2-ckpt results/P2_formal/best_model.pt
"""

import argparse, os, sys
from pathlib import Path
import numpy as np
import torch

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2-ckpt", type=str, default=str(_REPO / "results/P2_formal/best_model.pt"))
    parser.add_argument("--num-samples", type=int, default=4)
    args = parser.parse_args()

    print("=" * 60)
    print("P3 Inference Smoke: predict_action + current_mask")
    print(f"  P2 checkpoint: {args.p2_ckpt}")
    print("=" * 60)

    # ── Load VLA + P2 weights ────────────────────────────────
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

    p2_state = torch.load(args.p2_ckpt, map_location="cpu")
    if "model_state_dict" in p2_state:
        p2_state = p2_state["model_state_dict"]
    vla.load_state_dict(p2_state, strict=False)
    vla = vla.to("cuda")
    vla.eval()
    for p in vla.parameters():
        p.requires_grad_(False)
    print("P2 weights loaded.")

    # ── Dataloader ───────────────────────────────────────────
    from laravla.dataloader import build_dataloader
    loader = build_dataloader(OmegaConf.create({"datasets": {"vla_data": {
        "dataset_py": "spatial_cot_libero", "spatial_root": SPATIAL,
        "index_path": IDX, "cot_root": COT,
        "alignment_path": SPATIAL + "/cot_spatial_alignment.json",
        "enable_dynamic_mask": True, "per_device_batch_size": args.num_samples,
        "num_workers": 0, "state_dim": 7,
    }}}), dataset_py="spatial_cot_libero")
    batch = next(iter(loader))
    B = len(batch)
    print(f"\nBatch size: {B}")

    images = [s["image"] for s in batch]
    instructions = [s["lang"] for s in batch]

    # Build current_mask from batch
    current_masks = np.stack([s["current_affordance_mask_agentview"] for s in batch])  # [B, 224, 224]

    # ── Test 1: predict_action WITHOUT mask (original path) ──
    print("\n[1] predict_action WITHOUT current_mask...")
    with torch.no_grad():
        out_no_mask = vla.predict_action(
            batch_images=images, instructions=instructions,
            state=np.stack([s["state"] for s in batch]),
        )
    print(f"  shape: {out_no_mask['normalized_actions'].shape}")
    assert out_no_mask['normalized_actions'].shape == (B, 8, 7)
    print("  ✅ Original inference works")

    # ── Test 2: predict_action WITH current_mask ─────────────
    print("\n[2] predict_action WITH current_mask...")
    with torch.no_grad():
        out_with_mask = vla.predict_action(
            batch_images=images, instructions=instructions,
            state=np.stack([s["state"] for s in batch]),
            current_masks=current_masks,
        )
    print(f"  shape: {out_with_mask['normalized_actions'].shape}")
    assert out_with_mask['normalized_actions'].shape == (B, 8, 7)
    assert "transition_tokens" in out_with_mask, "Missing transition_tokens!"
    print(f"  transition_tokens shape: {out_with_mask['transition_tokens'].shape}")
    print("  ✅ Transition-conditioned inference works")

    # ── Test 3: compare with vs without mask ─────────────────
    print("\n[3] Comparing with vs without mask...")
    diff = np.abs(out_no_mask['normalized_actions'] - out_with_mask['normalized_actions'])
    print(f"  Mean abs diff: {diff.mean():.6f}")
    print(f"  Max abs diff:  {diff.max():.6f}")
    print(f"  (With gate≈0.09, transition should moderately modulate actions)")

    # ── Test 4: no NaN ──────────────────────────────────────
    assert not np.isnan(out_with_mask['normalized_actions']).any(), "NaN in actions!"
    print("\n[4] No NaN ✅")

    print("\n" + "=" * 60)
    print("P3 INFERENCE SMOKE COMPLETE — ALL PASS ✅")
    print("=" * 60)
    print(f"\nNext: LIBERO rollout evaluation with P2 checkpoint.")
    print(f"  vla.predict_action(images, instructions, current_masks=GT_masks)")


if __name__ == "__main__":
    main()
