#!/usr/bin/env python
"""
Diagnose where transition token collapse happens.
==================================================
Checks token cosine at 3 stages:
  1. After input type_embedding (q_init → before transition_module)
  2. After transition_module output (z_raw)
  3. After output type_embedding (z_head → before heads)

Also checks target mask similarity (C-F, C-G, F-G IoU).

Usage:
    python tools/debug_latent_slots.py
"""

import os, sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

CKPT = os.environ.get("LARAVLA_CKPT",
    str(_REPO.parent / "models/LaRA-VLA-libero/checkpoints/steps_25000_pytorch_model.pt"))
SPATIAL = str(_REPO / "output" / "spatial_lara_libero")
V4_IDX = str(_REPO / "output" / "spatial_lara_libero_no_noops" /
             "spatial_lara_libero_index_cot_transition_all_fixed_v4_tau.jsonl")
COT = os.environ.get("LEROBOT_ROOT",
    str(_REPO.parent / "datasets" / "lovejuly" / "libero_lerobot_all"))


def cos_matrix(z):
    """Compute pairwise cosine matrix [T,T] from [B,T,D]."""
    z_n = F.normalize(z.float(), dim=-1)
    z_mean = z_n.mean(dim=0)  # [T, D]
    return z_mean @ z_mean.T


def inter_intra_cos(z):
    """(inter_type_cos, intra_type_cos) for [B, 6, D]."""
    sim = cos_matrix(z)  # [6, 6]
    inter, intra = [], []
    for i in range(6):
        ti = 0 if i < 2 else (1 if i < 4 else (2 if i < 5 else 3))
        for j in range(i + 1, 6):
            tj = 0 if j < 2 else (1 if j < 4 else (2 if j < 5 else 3))
            if ti == tj:
                intra.append(sim[i, j])
            else:
                inter.append(sim[i, j])
    inter_m = torch.stack(inter).mean() if inter else torch.tensor(0.0)
    intra_m = torch.stack(intra).mean() if intra else torch.tensor(0.0)
    return inter_m.item(), intra_m.item()


def mask_iou(a, b):
    """IoU between two binary masks [H,W]."""
    a_bool, b_bool = a > 0.5, b > 0.5
    inter = (a_bool & b_bool).sum()
    union = (a_bool | b_bool).sum()
    return (inter / max(union, 1)).item()


def main():
    print("=" * 60)
    print("Latent Slot Collapse Diagnostic")
    print("=" * 60)

    # ── Load model ─────────────────────────────────────────
    from laravla.model.tools import read_mode_config
    from omegaconf import OmegaConf
    model_cfg, _ = read_mode_config(Path(CKPT))
    model_cfg["framework"]["mask_conditioned_transition"] = {
        "enable": True, "num_mask_tokens": 8, "num_transition_tokens": 6,
        "mask_res": 56, "num_relation_labels": 6, "transition_dim": 512,
        "loss_weights": {"future_mask": 0.05, "goal_mask": 0.10, "relation": 0.05,
                         "current_mask": 0.05, "dino_future": 0.05},
        "dino": {"model_name": "dinov2_vitb14", "dino_dim": 768, "num_patches": 256},
    }
    from laravla.model.framework import build_framework
    vla = build_framework(OmegaConf.create(model_cfg))
    vla.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)
    vla = vla.to("cuda")
    vla.eval()
    print("Model loaded.")

    tm = vla.transition_module
    type_emb = tm.type_embedding.data  # [1, 6, 512]

    # ── Load a few real samples ─────────────────────────────
    from laravla.dataloader import build_dataloader
    loader = build_dataloader(OmegaConf.create({"datasets": {"vla_data": {
        "dataset_py": "spatial_cot_libero", "spatial_root": SPATIAL,
        "index_path": V4_IDX, "cot_root": COT,
        "alignment_path": SPATIAL + "/cot_spatial_alignment.json",
        "enable_dynamic_mask": True, "per_device_batch_size": 8,
        "num_workers": 0, "state_dim": 7,
    }}}), dataset_py="spatial_cot_libero")
    batch = next(iter(loader))
    print(f"Batch: {len(batch)} samples")

    # ── Target mask similarity ──────────────────────────────
    print(f"\n{'='*60}")
    print("TARGET MASK SIMILARITY (are targets different enough?)")
    print(f"{'='*60}")
    cur_masks = np.stack([s["current_affordance_mask_agentview"] for s in batch])
    fut_masks = np.stack([s.get("future_tau_mask_agentview",
        s.get("future_affordance_mask_agentview")) for s in batch])
    gl_masks = np.stack([s["goal_affordance_mask_agentview"] for s in batch])

    cf_ious = [mask_iou(cur_masks[i], fut_masks[i]) for i in range(len(batch))]
    cg_ious = [mask_iou(cur_masks[i], gl_masks[i]) for i in range(len(batch))]
    fg_ious = [mask_iou(fut_masks[i], gl_masks[i]) for i in range(len(batch))]

    print(f"  C-F IoU: mean={np.mean(cf_ious):.3f}  min={np.min(cf_ious):.3f}  max={np.max(cf_ious):.3f}")
    print(f"  C-G IoU: mean={np.mean(cg_ious):.3f}  min={np.min(cg_ious):.3f}  max={np.max(cg_ious):.3f}")
    print(f"  F-G IoU: mean={np.mean(fg_ious):.3f}  min={np.min(fg_ious):.3f}  max={np.max(fg_ious):.3f}")

    target_diverse = np.mean(cf_ious) < 0.7 or np.mean(cg_ious) < 0.5
    if target_diverse:
        print("  → Targets ARE diverse. If tokens still collapse, it's a structural issue.")
    else:
        print("  → Targets are SIMILAR! Token collapse may not hurt task — "
              "consider target-aware diversity.")

    # ── Token cosine at 3 stages ────────────────────────────
    print(f"\n{'='*60}")
    print("TOKEN COSINE AT 3 STAGES")
    print(f"{'='*60}")

    images = [s["image"] for s in batch]
    instructions = [s["lang"] for s in batch]

    with torch.no_grad():
        qwen_out = vla.qwen_vl_interface.encode_observation(
            images=images, instructions=instructions, output_hidden_states=True)
        vlm_hidden = qwen_out.hidden_states[-1].float()  # [B, L, 2560]

    vlm_proj = vla.vlm_projector(vlm_hidden)

    # --- Stage 1: query + input type embedding ---
    B = vlm_proj.shape[0]
    q_init = tm.transition_queries.expand(B, -1, -1) + tm.type_embedding.expand(B, -1, -1)
    q_init_inter, q_init_intra = inter_intra_cos(q_init)
    print(f"  Stage 1 (q + input type_emb):      inter={q_init_inter:.3f}  intra={q_init_intra:.3f}")

    # --- Stage 2: after transition_module ---
    z_raw = tm(vlm_proj, mask_tokens=None)  # [B, 6, 512]
    z_raw_inter, z_raw_intra = inter_intra_cos(z_raw)
    print(f"  Stage 2 (after transition_module):  inter={z_raw_inter:.3f}  intra={z_raw_intra:.3f}")

    # --- Stage 3: after output type embedding ---
    z_head = z_raw + tm.type_embedding
    z_head_inter, z_head_intra = inter_intra_cos(z_head)
    print(f"  Stage 3 (z + output type_emb):      inter={z_head_inter:.3f}  intra={z_head_intra:.3f}")

    # --- Check type_embedding magnitude vs token magnitude ---
    type_norm = type_emb.float().norm(dim=-1).mean().item()  # mean over 6 tokens
    token_norm = z_raw.float().norm(dim=-1).mean().item()
    print(f"\n  type_embedding norm: {type_norm:.2f}")
    print(f"  token norm:          {token_norm:.2f}")
    print(f"  ratio (type/token):  {type_norm/token_norm:.4f}")

    # --- Full per-token cosine matrix ---
    print(f"\n  Stage 2 cosine matrix [6x6]:")
    sim_raw = cos_matrix(z_raw)
    for i in range(6):
        row = " ".join(f"{sim_raw[i,j].item():.3f}" for j in range(6))
        print(f"    t{i}: {row}")

    # --- Where does collapse happen? ---
    print(f"\n{'='*60}")
    print("DIAGNOSIS")
    print(f"{'='*60}")
    if q_init_inter < 0.7 and z_raw_inter > 0.9:
        print("  ⚠️  Transition module WASHES OUT slot identity.")
        print("     q_init is diverse → z_raw is collapsed.")
        print("     FIX: strengthen type identity residual after transition.")
    elif q_init_inter > 0.9:
        print("  ⚠️  Type embedding too weak at input.")
        print("     FIX: increase type_embedding scale or init std.")
    elif z_raw_inter > 0.9 and z_head_inter < 0.7:
        print("  ✅ Output type embedding restores diversity!")
        print("     The issue is that diagnostics were measuring pre-head tokens.")
    else:
        print(f"  q_init={q_init_inter:.3f} → z_raw={z_raw_inter:.3f} → z_head={z_head_inter:.3f}")

    if target_diverse and z_head_inter > 0.9:
        print("  🔴 CRITICAL: Targets are diverse but tokens still collapse at heads.")
        print("     Need SLOT IDENTITY RESIDUAL (v4).")


if __name__ == "__main__":
    main()
