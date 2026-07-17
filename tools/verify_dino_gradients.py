#!/usr/bin/env python
"""
Verify DINO head gradients + save-reload consistency.
======================================================
Steps 3-5 of the 8-step DINO fix plan.

3. Verify dino_future_head/transition/vlm_projector/slot_queries all get grads
4. Verify parameters change after optimizer step
5. Save-reload test: pred before vs after must be identical
"""

import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np
import torch

_REPO = Path(".").resolve()
CKPT = os.environ.get("LARAVLA_CKPT",
    str(_REPO.parent / "models/LaRA-VLA-libero/checkpoints/steps_25000_pytorch_model.pt"))
SPATIAL = "output/spatial_lara_libero"
IDX = "output/spatial_lara_libero_no_noops/spatial_lara_libero_index_cot_transition_all_fixed_v4_tau.jsonl"
COT = os.environ.get("LEROBOT_ROOT",
    str(_REPO.parent / "datasets/lovejuly/libero_lerobot_all"))


def main():
    print("=" * 60)
    print("DINO Gradient + Save-Reload Verification")
    print("=" * 60)

    from laravla.model.tools import read_mode_config
    from omegaconf import OmegaConf
    model_cfg, _ = read_mode_config(Path(CKPT))
    model_cfg["framework"]["mask_conditioned_transition"] = {
        "enable": True, "num_transition_tokens": 6, "mask_res": 56,
        "num_relation_labels": 7, "transition_dim": 512,
        "loss_weights": {"future_mask":0.05,"goal_mask":0.10,"relation":0.05,
                         "current_mask":0.05,"dino_future":0.05,"slot_residual_gamma":1.5},
        "dino": {"model_name":"dinov2_vitb14","dino_dim":768,"num_patches":256},
    }
    from laravla.model.framework import build_framework
    vla = build_framework(OmegaConf.create(model_cfg))
    vla.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)
    from laravla.model.modules.spatial_transition import (
        SpatialTransitionBackbone, P1NoMaskWrapper, dino_future_cosine
    )
    backbone = SpatialTransitionBackbone(vlm_dim=2560, hidden_dim=512, num_slots=6, gamma=1.5)
    backbone = backbone.to("cuda"); backbone.train()
    vla = vla.to("cuda")

    # Get one batch
    from laravla.dataloader import build_dataloader
    loader = build_dataloader(OmegaConf.create({"datasets": {"vla_data": {
        "dataset_py": "spatial_cot_libero", "spatial_root": SPATIAL, "index_path": IDX,
        "cot_root": COT, "alignment_path": SPATIAL + "/cot_spatial_alignment.json",
        "enable_dynamic_mask": True, "per_device_batch_size": 4,
        "num_workers": 0, "state_dim": 7,
    }}}), dataset_py="spatial_cot_libero")
    batch = next(iter(loader))
    images = [s["image"] for s in batch]

    with torch.no_grad():
        qo = vla.qwen_vl_interface.encode_observation(
            images=images, instructions=[s["lang"] for s in batch],
            output_hidden_states=True)
        vlm_hidden = qo.hidden_states[-1].clone()
        cur_rgb = torch.stack([torch.from_numpy(s["image_current_raw"]).permute(2,0,1) for s in batch]).to("cuda")
        fut_rgb = torch.stack([torch.from_numpy(s["image_tau_future_raw"]).permute(2,0,1) for s in batch]).to("cuda")
        dino_target = vla.dino_encoder(fut_rgb).clone()

    # ── Step 3: Gradient verification ───────────────────────
    print("\n[Step 3] Gradient check...")
    opt = torch.optim.AdamW(backbone.parameters(), lr=3e-4)
    opt.zero_grad(set_to_none=True)

    out = backbone(vlm_hidden)
    result = dino_future_cosine(out.pred_future_dino, dino_target)
    loss = result["loss"]
    loss.backward()

    modules_to_check = {
        "dino_future_head": backbone.dino_future_head,
        "transition_module": backbone.transition_module,
        "vlm_projector": backbone.vlm_projector,
        "slot_queries": backbone.slot_queries,
    }
    all_good = True
    for name, mod in modules_to_check.items():
        if isinstance(mod, torch.Tensor):
            grad_norm = mod.grad.norm().item() if mod.grad is not None else 0
        else:
            grad_norm = sum(p.grad.norm().item() for p in mod.parameters() if p.grad is not None)
        ok = grad_norm > 0
        print(f"  {name}: grad_norm={grad_norm:.4f} {'✅' if ok else '❌ NO GRADIENT'}")
        if not ok: all_good = False

    assert all_good, "Some modules have zero gradient!"

    # ── Step 4: Parameter delta after step ──────────────────
    print("\n[Step 4] Parameter delta after optimizer.step()...")
    before_params = {name: p.clone() for name, p in backbone.named_parameters() if p.requires_grad}
    opt.step()
    deltas = {}
    for name, p in backbone.named_parameters():
        if name in before_params:
            delta = (p - before_params[name]).abs().max().item()
            deltas[name] = delta
    max_delta = max(deltas.values())
    print(f"  Max param delta: {max_delta:.2e} {'✅' if max_delta > 0 else '❌'}")
    assert max_delta > 0, "Parameters unchanged after optimizer step!"

    # ── Step 5: Save-reload test ────────────────────────────
    print("\n[Step 5] Save-reload consistency...")
    # Save pred before
    backbone.eval()
    with torch.no_grad():
        out_before = backbone(vlm_hidden)
        pred_before = out_before.pred_future_dino.clone()

    # Save checkpoint
    state = backbone.state_dict()
    # Reload
    backbone2 = SpatialTransitionBackbone(vlm_dim=2560, hidden_dim=512, num_slots=6, gamma=1.5)
    backbone2.load_state_dict(state, strict=True)
    backbone2 = backbone2.to("cuda")
    backbone2.eval()

    # Forward with same inputs
    with torch.no_grad():
        out_after = backbone2(vlm_hidden)
        pred_after = out_after.pred_future_dino

    pred_diff = (pred_before - pred_after).abs().max().item()
    cos_before = torch.nn.functional.cosine_similarity(
        pred_before.flatten(1).float(), dino_target.flatten(1).float()).mean().item()
    cos_after = torch.nn.functional.cosine_similarity(
        pred_after.flatten(1).float(), dino_target.flatten(1).float()).mean().item()

    print(f"  pred max_abs_diff: {pred_diff:.2e} {'✅' if pred_diff < 1e-5 else '❌'}")
    print(f"  cos before: {cos_before:.4f}  cos after: {cos_after:.4f}  diff: {abs(cos_before-cos_after):.2e}")
    assert pred_diff < 1e-5, f"Prediction differs after save-reload: {pred_diff:.2e}"
    assert abs(cos_before - cos_after) < 1e-6, "Cosine differs after save-reload"

    print(f"\n{'='*60}")
    print("✅ Steps 3-5 PASSED: gradients OK, params update, save-reload consistent")


if __name__ == "__main__":
    main()
