#!/usr/bin/env python
"""
P2–P1 exact parity test: same batch, compare forward outputs.
==============================================================
Ensures P2's internal P1 forward produces identical results to the
original P1NoMaskWrapper forward (same z_student, mask logits,
relation logits, pred_future_dino).

Usage:
    python tools/debug_p2_p1_parity.py
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
P1_CKPT = str(_REPO / "results/P1_step5_nodistill_2k/best_student.pt")
SPATIAL = str(_REPO / "output/spatial_lara_libero")
IDX = str(_REPO / "output/spatial_lara_libero_no_noops" /
          "spatial_lara_libero_index_cot_transition_all_fixed_v4_tau.jsonl")
COT = os.environ.get("LEROBOT_ROOT",
    str(_REPO.parent / "datasets" / "lovejuly" / "libero_lerobot_all"))


def inter_intra_cos(z):
    zn = F.normalize(z.float(), dim=-1).mean(dim=0); s = zn @ zn.T
    inter, intra = [], []
    for i in range(6):
        ti = 0 if i<2 else (1 if i<4 else (2 if i<5 else 3))
        for j in range(i+1,6):
            tj = 0 if j<2 else (1 if j<4 else (2 if j<5 else 3))
            (intra if ti==tj else inter).append(s[i,j])
    return (torch.stack(inter).mean().item() if inter else 0,
            torch.stack(intra).mean().item() if intra else 0)


def main():
    print("=" * 60)
    print("P2–P1 Exact Parity Test")
    print("=" * 60)

    # ── Build P1NoMaskWrapper (reference) ────────────────────
    from laravla.model.tools import read_mode_config
    from omegaconf import OmegaConf
    model_cfg, _ = read_mode_config(Path(CKPT))
    model_cfg["framework"]["mask_conditioned_transition"] = {
        "enable": True, "num_mask_tokens": 8, "num_transition_tokens": 6,
        "mask_res": 56, "num_relation_labels": 7, "transition_dim": 512,
        "loss_weights": {"future_mask": 0.05, "goal_mask": 0.10, "relation": 0.05,
                         "current_mask": 0.05, "dino_future": 0.05,
                         "slot_residual_gamma": 1.5},
        "dino": {"model_name": "dinov2_vitb14", "dino_dim": 768, "num_patches": 256},
    }
    from laravla.model.framework import build_framework
    vla = build_framework(OmegaConf.create(model_cfg))
    vla.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)

    # Load P1 into VLA (same as P2 training script)
    p1_state = torch.load(P1_CKPT, map_location="cpu")
    if "p1_state_dict" in p1_state:
        p1_state = p1_state["p1_state_dict"]
    remapped = {}
    for k, v in p1_state.items():
        if k.startswith("mask_decoder."):
            remapped[k.replace("mask_decoder.", "current_mask_decoder.")] = v
        else:
            remapped[k] = v
    vla.load_state_dict(remapped, strict=False)
    for k, v in remapped.items():
        if k.startswith("current_mask_decoder."):
            for prefix in ["future_mask_decoder.", "goal_mask_decoder."]:
                vla.state_dict()[k.replace("current_mask_decoder.", prefix)].copy_(v)
    vla = vla.to("cuda")

    # ── Build P1NoMaskWrapper (reference) ────────────────────
    from laravla.model.modules.spatial_transition import P1NoMaskWrapper
    p1_wrapper = P1NoMaskWrapper(vla).to("cuda")
    p1_wrapper.load_state_dict(p1_state, strict=False)
    p1_wrapper.eval()
    for p in p1_wrapper.parameters():
        p.requires_grad_(False)

    # ── Set up P2 forward ────────────────────────────────────
    vla.training_stage = "transition_action_nomask"
    vla.eval()

    # ── Load same batch ──────────────────────────────────────
    from laravla.dataloader import build_dataloader
    loader = build_dataloader(OmegaConf.create({"datasets": {"vla_data": {
        "dataset_py": "spatial_cot_libero", "spatial_root": SPATIAL,
        "index_path": IDX, "cot_root": COT,
        "alignment_path": SPATIAL + "/cot_spatial_alignment.json",
        "enable_dynamic_mask": True, "per_device_batch_size": 4,
        "num_workers": 0, "state_dim": 7,
    }}}), dataset_py="spatial_cot_libero")
    batch = next(iter(loader))

    # ── Verify target fields ─────────────────────────────────
    print("\nTarget parity:")
    for k in ["hdf5_frame_idx", "hdf5_tau_future_idx", "tau_future_valid"]:
        vals = [s.get(k, "N/A") for s in batch]
        print(f"  {k}: {vals}")

    # ── P1 wrapper forward ───────────────────────────────────
    images = [s["image"] for s in batch]
    instructions = [s["lang"] for s in batch]
    cur_masks = torch.from_numpy(
        np.stack([s["current_affordance_mask_agentview"] for s in batch])
    ).unsqueeze(1).to("cuda").float()
    future_masks = torch.from_numpy(
        np.stack([s.get("future_tau_mask_agentview",
                        s.get("future_affordance_mask_agentview",
                              np.zeros((224,224), dtype=np.float32))) for s in batch])
    ).to("cuda").float()
    goal_masks = torch.from_numpy(
        np.stack([s["goal_affordance_mask_agentview"] for s in batch])
    ).to("cuda").float()
    rel_ids = torch.tensor([s["relation_label_id"] for s in batch],
                           dtype=torch.long, device="cuda")

    # DINO targets
    cur_tensors = [torch.from_numpy(np.array(s["image"][0], dtype=np.uint8)).permute(2,0,1) for s in batch]
    cur_rgb = torch.stack(cur_tensors).to("cuda")
    fut_imgs = [s.get("image_tau_future", s.get("image_next", None)) for s in batch]
    fut_tensors = []
    for fi in fut_imgs:
        if fi is not None and isinstance(fi, list) and len(fi) > 0: fi = fi[0]
        if fi is not None:
            fut_tensors.append(torch.from_numpy(np.array(fi, dtype=np.uint8)).permute(2,0,1))
        else:
            fut_tensors.append(cur_tensors[len(fut_tensors)])
    fut_rgb = torch.stack(fut_tensors).to("cuda")

    with torch.no_grad():
        qwen_out = vla.qwen_vl_interface.encode_observation(
            images=images, instructions=instructions, output_hidden_states=True)
        vlm_hidden = qwen_out.hidden_states[-1]
        dino_cur = vla.dino_encoder(cur_rgb)
        dino_fut = vla.dino_encoder(fut_rgb)

    # P1 wrapper forward (reference)
    with torch.no_grad():
        p1_out = p1_wrapper(vlm_hidden, cur_masks, future_masks, goal_masks, rel_ids,
                            dino_future_target=dino_fut, dino_cur=dino_cur)

    # P2 internal forward (via VLA)
    with torch.no_grad():
        p2_out = vla.forward(batch)

    # ── Compare ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Exact comparison (P1 wrapper vs P2 VLA forward)")
    print("=" * 60)

    # Mask losses
    for name in ["current_mask_loss", "future_mask_loss", "goal_mask_loss", "relation_loss"]:
        p1v = p1_out.get(name, torch.tensor(float("nan"))).item()
        p2v = p2_out.get(name, torch.tensor(float("nan"))).item()
        diff = abs(p1v - p2v)
        flag = "✅" if diff < 0.01 else "❌"
        print(f"  {name}: P1={p1v:.4f}  P2={p2v:.4f}  diff={diff:.4f}  {flag}")

    # DINO
    for name in ["dino_future_loss", "dino_future_cos"]:
        p1v = p1_out.get(name, torch.tensor(float("nan"))).item()
        p2v = p2_out.get(name, torch.tensor(float("nan"))).item()
        print(f"  {name}: P1={p1v:.4f}  P2={p2v:.4f}")

    # Transition tokens
    z_p1 = p1_out["transition_tokens"]
    z_p2 = p2_out.get("transition_tokens")
    if z_p2 is not None:
        max_diff = (z_p1 - z_p2).abs().max().item()
        print(f"  z_student max_abs_diff: {max_diff:.2e} {'✅' if max_diff < 1e-5 else '❌'}")

    # ── Train mode effect ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Train mode effect on frozen P1")
    print("=" * 60)
    vla.train()
    with torch.no_grad():
        p2_train_out = vla.forward(batch)
    for name in ["current_mask_loss", "future_mask_loss", "goal_mask_loss"]:
        ev = p2_out.get(name, torch.tensor(float("nan"))).item()
        tv = p2_train_out.get(name, torch.tensor(float("nan"))).item()
        print(f"  {name}: eval={ev:.4f}  train={tv:.4f}  Δ={abs(ev-tv):.4f}")
    vla.eval()

    # ── Check DINO nan ───────────────────────────────────────
    print("\nDINO check:")
    p1_pred_dino = p1_out.get("dino_future_loss", None)
    print(f"  dino_future_target norm: {dino_fut.float().norm():.2f}")
    print(f"  dino_future_target finite: {torch.isfinite(dino_fut).all().item()}")
    future_tokens = p1_out["transition_tokens"][:, 2:4, :]
    pred = vla.dino_future_head(future_tokens)
    print(f"  pred_future_dino norm: {pred.float().norm():.2f}")
    print(f"  pred finite: {torch.isfinite(pred).all().item()}")
    cos_sim = (F.normalize(pred.float(), dim=-1) * F.normalize(dino_fut.float(), dim=-1)).sum(dim=-1).mean().item()
    print(f"  DINO cos (manual): {cos_sim:.4f}")

    # ── P1-relevant missing keys ─────────────────────────────
    print("\nP1-relevant checkpoint key check:")
    required_prefixes = ["vlm_projector.", "transition_module.", "current_mask_decoder.",
                         "relation_head.", "dino_future_head."]
    remapped_keys = list(remapped.keys())
    for prefix in required_prefixes:
        count = sum(1 for k in remapped_keys if k.startswith(prefix))
        print(f"  {prefix}: {count} keys loaded")


if __name__ == "__main__":
    main()
