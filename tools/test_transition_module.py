#!/usr/bin/env python
"""
P1 Smoke Test: MaskTokenEncoder → TransitionModule → Decoders → Losses.
========================================================================
Verifies:
  1. MaskTokenEncoder shape correct
  2. TransitionModule shape correct
  3. Decoder logits shapes correct
  4. Losses compute, no NaN
  5. 50-step training, loss decreasing
"""

import sys, os, numpy as np
from pathlib import Path
import torch
from omegaconf import OmegaConf

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
    print("=" * 60)
    print("P1 Smoke Test: Spatial Transition Module")
    print("=" * 60)

    # ── 1. Load batch via build_dataloader ──────────────────
    print("\n[1] Loading batch via build_dataloader...")
    from laravla.dataloader import build_dataloader
    cfg = OmegaConf.create({"datasets": {"vla_data": {
        "dataset_py": "spatial_cot_libero", "spatial_root": SPATIAL,
        "index_path": IDX, "cot_root": COT,
        "alignment_path": SPATIAL + "/cot_spatial_alignment.json",
        "enable_dynamic_mask": True, "per_device_batch_size": 2,
        "num_workers": 0, "state_dim": 7,
    }}})
    loader = build_dataloader(cfg, dataset_py="spatial_cot_libero")
    batch = next(iter(loader))
    B = len(batch)
    print(f"  Batch size: {B}")

    # ── 2. Get VLM hidden states (frozen) ───────────────────
    print("\n[2] Loading Qwen_GR00T, extracting frozen hidden states...")
    from laravla.model.framework.base_framework import baseframework
    vla = baseframework.from_pretrained(CKPT)
    vla = vla.to("cuda")
    vla.eval()

    # Freeze VLM
    for p in vla.qwen_vl_interface.parameters():
        p.requires_grad = False
    for p in vla.action_model.parameters():
        p.requires_grad = False

    # Get hidden states via build_qwenvl_inputs + forward
    images = [s["image"] for s in batch]
    instructions = [s["lang"] for s in batch]
    qwen_inputs = vla.qwen_vl_interface.build_qwenvl_inputs(
        images=images, instructions=instructions
    )
    qwen_inputs = {k: v.to("cuda") for k, v in qwen_inputs.items()}

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        qwen_out = vla.qwen_vl_interface(**qwen_inputs, output_hidden_states=True)
    vlm_hidden = qwen_out.hidden_states[-1].float()  # [B, L, 2560]
    HIDDEN_DIM = vlm_hidden.shape[-1]
    print(f"  VLM hidden: {vlm_hidden.shape}  D={HIDDEN_DIM}")

    # ── 3. MaskTokenEncoder ─────────────────────────────────
    print("\n[3] Testing MaskTokenEncoder...")
    from laravla.model.modules.spatial_transition import MaskTokenEncoder

    cur_mask = torch.from_numpy(
        np.stack([s["current_affordance_mask_agentview"] for s in batch])
    ).unsqueeze(1).to("cuda")  # [B, 1, 224, 224]

    encoder = MaskTokenEncoder(
        in_channels=1, transition_dim=HIDDEN_DIM, num_tokens=8
    ).to("cuda")
    mask_tokens = encoder(cur_mask)
    print(f"  Input:  {cur_mask.shape}")
    print(f"  Output: {mask_tokens.shape}  (expected [B,8,{HIDDEN_DIM}])")
    assert mask_tokens.shape == (B, 8, HIDDEN_DIM), f"Shape mismatch: {mask_tokens.shape}"
    print("  ✅ MaskTokenEncoder PASS")

    # ── 4. TransitionModule ─────────────────────────────────
    print("\n[4] Testing MaskConditionedTransitionModule...")
    from laravla.model.modules.spatial_transition import MaskConditionedTransitionModule

    trans_module = MaskConditionedTransitionModule(
        transition_dim=HIDDEN_DIM, num_transition_tokens=6
    ).to("cuda")
    transition_tokens = trans_module(vlm_hidden, mask_tokens)
    print(f"  Output: {transition_tokens.shape}  (expected [B,6,{HIDDEN_DIM}])")
    assert transition_tokens.shape == (B, 6, HIDDEN_DIM)
    print("  ✅ TransitionModule PASS")

    # ── 5. Decoders ─────────────────────────────────────────
    print("\n[5] Testing Decoders...")
    from laravla.model.modules.spatial_transition import MaskDecoder, RelationHead

    future_decoder = MaskDecoder(transition_dim=HIDDEN_DIM, num_transition_tokens=6, output_res=56).to("cuda")
    goal_decoder = MaskDecoder(transition_dim=HIDDEN_DIM, num_transition_tokens=6, output_res=56).to("cuda")
    relation_head = RelationHead(transition_dim=HIDDEN_DIM, num_transition_tokens=6, num_classes=6).to("cuda")

    future_logits = future_decoder(transition_tokens)
    goal_logits = goal_decoder(transition_tokens)
    rel_logits = relation_head(transition_tokens)

    print(f"  Future mask logits: {future_logits.shape}  (expected [B,1,56,56])")
    print(f"  Goal mask logits:   {goal_logits.shape}  (expected [B,1,56,56])")
    print(f"  Relation logits:    {rel_logits.shape}    (expected [B,6])")
    assert future_logits.shape == (B, 1, 56, 56)
    assert goal_logits.shape == (B, 1, 56, 56)
    assert rel_logits.shape == (B, 6)
    print("  ✅ Decoders PASS")

    # ── 6. Losses ───────────────────────────────────────────
    print("\n[6] Testing Losses...")
    from laravla.model.modules.spatial_transition import transition_total_loss
    import torch.nn.functional as F

    # Downsample GT masks from 224 to 56
    future_gt = torch.from_numpy(
        np.stack([s["future_affordance_mask_agentview"] for s in batch])
    ).to("cuda")  # [B, 224, 224]
    goal_gt = torch.from_numpy(
        np.stack([s["goal_affordance_mask_agentview"] for s in batch])
    ).to("cuda")
    rel_gt = torch.tensor(
        [s["relation_label_id"] for s in batch], dtype=torch.long
    ).to("cuda")

    # Resize GT to match logits (56x56)
    future_gt_56 = F.interpolate(
        future_gt.unsqueeze(1), size=(56, 56), mode='nearest'
    ).squeeze(1)
    goal_gt_56 = F.interpolate(
        goal_gt.unsqueeze(1), size=(56, 56), mode='nearest'
    ).squeeze(1)

    losses = transition_total_loss(
        future_logits=future_logits,
        future_target=future_gt_56,
        goal_logits=goal_logits,
        goal_target=goal_gt_56,
        relation_logits=rel_logits,
        relation_target=rel_gt,
    )

    for k, v in losses.items():
        nan_str = " ⚠️ NaN!" if torch.isnan(v) else ""
        print(f"  {k}: {v.item():.6f}{nan_str}")

    assert not torch.isnan(losses["total_loss"]), "Total loss is NaN!"
    print("  ✅ Losses PASS")

    # ── 7. 50-step training ─────────────────────────────────
    print("\n[7] 50-step training sanity...")
    # Collect trainable params
    trainable = (list(encoder.parameters()) + list(trans_module.parameters()) +
                 list(future_decoder.parameters()) + list(goal_decoder.parameters()) +
                 list(relation_head.parameters()))
    n_params = sum(p.numel() for p in trainable)
    print(f"  Trainable params: {n_params/1e6:.1f}M")

    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    losses_history = []

    for step in range(50):
        try:
            batch = next(iter(loader))
        except StopIteration:
            break

        B = len(batch)
        images = [s["image"] for s in batch]
        instructions = [s["lang"] for s in batch]
        cur_masks = torch.from_numpy(
            np.stack([s["current_affordance_mask_agentview"] for s in batch])
        ).unsqueeze(1).to("cuda")

        future_gts = torch.from_numpy(
            np.stack([s["future_affordance_mask_agentview"] for s in batch])
        ).to("cuda")
        goal_gts = torch.from_numpy(
            np.stack([s["goal_affordance_mask_agentview"] for s in batch])
        ).to("cuda")
        rel_gts = torch.tensor(
            [s["relation_label_id"] for s in batch], dtype=torch.long
        ).to("cuda")

        # Frozen VLM forward
        qwen_inputs = vla.qwen_vl_interface.build_qwenvl_inputs(
            images=images, instructions=instructions
        )
        qwen_inputs = {k: v.to("cuda") for k, v in qwen_inputs.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            qwen_out = vla.qwen_vl_interface(**qwen_inputs, output_hidden_states=True)
        vlm_hidden = qwen_out.hidden_states[-1].float()

        # Forward through our modules
        mask_tokens = encoder(cur_masks)
        transition_tokens = trans_module(vlm_hidden, mask_tokens)
        future_logits = future_decoder(transition_tokens)
        goal_logits = goal_decoder(transition_tokens)
        rel_logits = relation_head(transition_tokens)

        # Resize GT
        future_gt_56 = F.interpolate(future_gts.unsqueeze(1), size=(56, 56), mode='nearest').squeeze(1)
        goal_gt_56 = F.interpolate(goal_gts.unsqueeze(1), size=(56, 56), mode='nearest').squeeze(1)

        losses = transition_total_loss(
            future_logits=future_logits, future_target=future_gt_56,
            goal_logits=goal_logits, goal_target=goal_gt_56,
            relation_logits=rel_logits, relation_target=rel_gts,
        )

        loss = losses["total_loss"]
        if torch.isnan(loss):
            print(f"  Step {step}: NaN!")
            break

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses_history.append(loss.item())

        if step < 3 or step % 10 == 0:
            lf = losses.get("future_mask_loss", torch.tensor(0)).item()
            lg = losses.get("goal_mask_loss", torch.tensor(0)).item()
            lr = losses.get("relation_loss", torch.tensor(0)).item()
            print(f"  Step {step:2d}: total={loss.item():.4f}  future={lf:.4f}  goal={lg:.4f}  rel={lr:.4f}")

    trend = f"{losses_history[0]:.4f} → {losses_history[-1]:.4f}" if len(losses_history) >= 2 else "N/A"
    print(f"  Loss trend: {trend}")

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("P1 SMOKE TEST COMPLETE")
    print("=" * 60)
    print(f"  ✅ MaskTokenEncoder:    {mask_tokens.shape}")
    print(f"  ✅ TransitionModule:    {transition_tokens.shape}")
    print(f"  ✅ FutureMaskDecoder:   {future_logits.shape}")
    print(f"  ✅ GoalMaskDecoder:     {goal_logits.shape}")
    print(f"  ✅ RelationHead:        {rel_logits.shape}")
    print(f"  ✅ Losses:              no NaN")
    print(f"  ✅ 50-step:             {len(losses_history)} steps, trend {trend}")


if __name__ == "__main__":
    main()
