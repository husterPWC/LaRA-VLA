#!/usr/bin/env python
"""
Convert clean P1 checkpoint → unified SpatialTransitionBackbone format.
=========================================================================
Old checkpoint (P1NoMaskWrapper state_dict):
    vlm_projector.*, transition_module.*, mask_decoder.*,
    relation_head.*, dino_future_head.*, posterior_encoder.*

New checkpoint:
    {"spatial_backbone_state_dict": backbone.state_dict()}

Migration maps:
    mask_decoder.* → mask_decoder.*           (same name, shared decoder)
    relation_head.* → relation_head.*         (same)
    dino_future_head.* → dino_future_head.*   (same)
    slot queries → slot_queries               (new name)
    transition_module.transition_queries → slot_queries (copy)
    transition_module.type_embedding → slot_queries is separate now

Verification: compare old P1NoMaskWrapper vs new backbone on fixed batches.
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
OLD_P1 = str(_REPO / "results/P1_step5_clean/best_student.pt")
NEW_P1 = str(_REPO / "results/P1_formal_unified/best_spatial_backbone.pt")


def main():
    print("=" * 60)
    print("Convert clean P1 → unified spatial_backbone")
    print("=" * 60)

    Path(NEW_P1).parent.mkdir(parents=True, exist_ok=True)

    # ── Load old checkpoint ─────────────────────────────────
    old = torch.load(OLD_P1, map_location="cpu")
    if "p1_state_dict" not in old:
        raise ValueError("Old checkpoint missing p1_state_dict")
    old_state = old["p1_state_dict"]

    # ── Build new backbone, load via submodule state_dicts ─
    from laravla.model.modules.spatial_transition import SpatialTransitionBackbone
    backbone = SpatialTransitionBackbone(
        vlm_dim=2560, hidden_dim=512, num_slots=6, gamma=1.5,
    )
    # slot_norm defaults match old F.layer_norm
    backbone.slot_norm.weight.data.fill_(1.0)
    backbone.slot_norm.bias.data.zero_()
    backbone.eval()

    # Direct copy from old wrapper (built via VLA + load old state)
    from laravla.model.tools import read_mode_config
    from omegaconf import OmegaConf
    model_cfg, _ = read_mode_config(Path(CKPT))
    model_cfg["framework"]["mask_conditioned_transition"] = {
        "enable": True, "num_mask_tokens": 8, "num_transition_tokens": 6,
        "mask_res": 56, "num_relation_labels": 7, "transition_dim": 512,
        "loss_weights": {"future_mask":0.05,"goal_mask":0.10,"relation":0.05,
                         "current_mask":0.05,"dino_future":0.05,"slot_residual_gamma":1.5},
        "dino": {"model_name":"dinov2_vitb14","dino_dim":768,"num_patches":256},
    }
    from laravla.model.framework import build_framework
    vla = build_framework(OmegaConf.create(model_cfg))
    vla.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)
    from laravla.model.modules.spatial_transition import P1NoMaskWrapper
    old_p1 = P1NoMaskWrapper(vla)
    old_p1.load_state_dict(old_state, strict=False)
    old_p1.eval()

    # Submodule-level state_dict loading (avoids key nesting issues)
    backbone.vlm_projector.load_state_dict(old_p1.vlm_projector.state_dict())
    backbone.transition_module.load_state_dict(old_p1.transition_module.state_dict())
    backbone.mask_decoder.load_state_dict(old_p1.mask_decoder.state_dict())
    backbone.relation_head.load_state_dict(old_p1.relation_head.state_dict())
    backbone.dino_future_head.load_state_dict(old_p1.dino_future_head.state_dict())
    backbone.slot_queries.data.copy_(
        old_p1.transition_module.transition_queries.data +
        old_p1.transition_module.type_embedding.data)
    print(f"  Parameters copied via submodule state_dicts")

    # ── Save ───────────────────────────────────────────────
    torch.save({
        "format_version": 1,
        "config": {"gamma": 1.5, "num_slots": 6, "hidden_dim": 512,
                   "tau_offset": 8, "future_source": "fixed_tau"},
        "spatial_backbone_state_dict": backbone.state_dict(),
    }, NEW_P1)
    print(f"\n  Saved to: {NEW_P1}")

    # ── Verify: compare old P1NoMaskWrapper vs new backbone ─
    print(f"\n{'='*60}")
    print("Verification")
    print(f"{'='*60}")

    vla = vla.to("cuda")
    old_p1 = old_p1.to("cuda")
    backbone = backbone.to("cuda")
    backbone.eval()

    # Get one batch
    SPATIAL = str(_REPO / "output/spatial_lara_libero")
    IDX = str(_REPO / "output/spatial_lara_libero_no_noops" /
              "spatial_lara_libero_index_cot_transition_all_fixed_v4_tau.jsonl")
    COT = os.environ.get("LEROBOT_ROOT",
        str(_REPO.parent / "datasets" / "lovejuly" / "libero_lerobot_all"))
    from laravla.dataloader import build_dataloader
    loader = build_dataloader(OmegaConf.create({"datasets": {"vla_data": {
        "dataset_py": "spatial_cot_libero", "spatial_root": SPATIAL,
        "index_path": IDX, "cot_root": COT,
        "alignment_path": SPATIAL + "/cot_spatial_alignment.json",
        "enable_dynamic_mask": True, "per_device_batch_size": 4,
        "num_workers": 0, "state_dim": 7,
    }}}), dataset_py="spatial_cot_libero")
    batch = next(iter(loader))
    images = [s["image"] for s in batch]
    instructions = [s["lang"] for s in batch]

    with torch.no_grad():
        qo = vla.qwen_vl_interface.encode_observation(
            images=images, instructions=instructions, output_hidden_states=True)
        vlm_hidden = qo.hidden_states[-1]

    # Compare
    old_z = old_p1(vlm_hidden,
        torch.from_numpy(np.stack([s["current_affordance_mask_agentview"] for s in batch])).unsqueeze(1).to("cuda").float(),
        torch.from_numpy(np.stack([s.get("future_tau_mask_agentview", s.get("future_affordance_mask_agentview", np.zeros((224,224),dtype=np.float32))) for s in batch])).to("cuda").float(),
        torch.from_numpy(np.stack([s["goal_affordance_mask_agentview"] for s in batch])).to("cuda").float(),
        torch.tensor([s["relation_label_id"] for s in batch], dtype=torch.long, device="cuda"),
    )
    new_out = backbone(vlm_hidden)

    z_diff = (old_z["transition_tokens"] - new_out.z_student).abs().max().item()
    cur_diff = (old_p1.mask_decoder(old_z["transition_tokens"][:, 0:2, :]) - new_out.current_mask_logits).abs().max().item()
    fut_diff = (old_p1.mask_decoder(old_z["transition_tokens"][:, 2:4, :]) - new_out.future_mask_logits).abs().max().item()
    goal_diff = (old_p1.mask_decoder(old_z["transition_tokens"][:, 4:5, :]) - new_out.goal_mask_logits).abs().max().item()
    rel_diff = (old_p1.relation_head(old_z["transition_tokens"][:, 5:6, :]) - new_out.relation_logits).abs().max().item()
    dino_old = old_p1.dino_future_head(old_z["transition_tokens"][:, 2:4, :])
    dino_diff = (dino_old - new_out.pred_future_dino).abs().max().item()

    print(f"  z_student diff:       {z_diff:.2e}")
    print(f"  current_mask diff:    {cur_diff:.2e}")
    print(f"  future_mask diff:     {fut_diff:.2e}")
    print(f"  goal_mask diff:       {goal_diff:.2e}")
    print(f"  relation_logits diff: {rel_diff:.2e}")
    print(f"  pred_future_dino diff:{dino_diff:.2e}")

    all_ok = max(z_diff, cur_diff, fut_diff, goal_diff, rel_diff, dino_diff) < 1e-5
    if all_ok:
        print(f"\n  ✅ Migration successful — all diffs < 1e-5")
        # Save with verified weights
        torch.save({
            "format_version": 1,
            "config": {"gamma": 1.5, "num_slots": 6, "hidden_dim": 512,
                       "tau_offset": 8, "future_source": "fixed_tau"},
            "spatial_backbone_state_dict": backbone.state_dict(),
        }, NEW_P1)
        print(f"  Saved to: {NEW_P1}")
    else:
        print(f"\n  ❌ Migration failed — diffs too large")


if __name__ == "__main__":
    main()
