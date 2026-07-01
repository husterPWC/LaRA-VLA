"""
P1TransitionWrapper: lightweight wrapper for DDP-safe P1 training.
===================================================================
Only wraps the 6 P1 trainable modules (~26.5M). Qwen-VL and Action
model stay outside DDP — each rank loads its own frozen copy locally.

Usage in training script:
    vla = build_framework(cfg)
    vla.load_state_dict(...)
    # Freeze VLA
    for p in vla.parameters(): p.requires_grad_(False)
    # Build P1 wrapper (trainable)
    p1_model = P1TransitionWrapper(vla).to(device)
    # DDP only wraps p1_model
    p1_model, optimizer, loader = accelerator.prepare(p1_model, optimizer, loader)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class P1TransitionWrapper(nn.Module):
    """Wraps P1 trainable modules only. Qwen-VL stays outside for no_grad encode."""

    def __init__(self, vla):
        super().__init__()
        # Copy references to P1 trainable sub-modules
        self.vlm_projector = vla.vlm_projector
        self.mask_token_encoder = vla.mask_token_encoder
        self.transition_module = vla.transition_module
        self.future_mask_decoder = vla.future_mask_decoder
        self.goal_mask_decoder = vla.goal_mask_decoder
        self.relation_head = vla.relation_head

        # Cache loss weights and config
        self.loss_weights = vla.transition_loss_weights
        self.mask_res = 56

    def forward(self, vlm_hidden, cur_masks, future_masks, goal_masks, rel_ids):
        """
        Args:
            vlm_hidden:    [B, L, 2560] frozen Qwen-VL hidden states
            cur_masks:     [B, 1, H, W] current affordance mask
            future_masks:  [B, H, W]    future mask GT
            goal_masks:    [B, H, W]    goal mask GT
            rel_ids:       [B] LongTensor relation label ids

        Returns:
            dict with total_loss, future_mask_loss, goal_mask_loss, relation_loss,
                 future_dice, goal_dice, relation_acc, transition_tokens
        """
        from laravla.model.modules.spatial_transition import transition_total_loss

        # Bottleneck projection
        vlm_proj = self.vlm_projector(vlm_hidden.float())

        # Mask tokens
        mask_tokens = self.mask_token_encoder(cur_masks)

        # Transition
        transition_tokens = self.transition_module(vlm_proj, mask_tokens)

        # Decode
        future_logits = self.future_mask_decoder(transition_tokens)
        goal_logits = self.goal_mask_decoder(transition_tokens)
        rel_logits = self.relation_head(transition_tokens)

        # Resize GT masks
        R = future_logits.shape[-1]
        future_gt = F.interpolate(
            future_masks.unsqueeze(1), size=(R, R), mode='nearest'
        ).squeeze(1)
        goal_gt = F.interpolate(
            goal_masks.unsqueeze(1), size=(R, R), mode='nearest'
        ).squeeze(1)

        # Losses
        w = self.loss_weights
        losses = transition_total_loss(
            future_logits=future_logits, future_target=future_gt,
            goal_logits=goal_logits, goal_target=goal_gt,
            relation_logits=rel_logits, relation_target=rel_ids,
            w_future=w.get("future_mask", 0.05),
            w_goal=w.get("goal_mask", 0.10),
            w_relation=w.get("relation", 0.05),
        )

        # Metrics
        future_dice = self._dice(future_logits, future_gt)
        goal_dice = self._dice(goal_logits, goal_gt)
        rel_pred = rel_logits.argmax(dim=1)
        valid = (rel_ids >= 0) & (rel_ids < rel_logits.shape[1])
        rel_acc = (rel_pred[valid] == rel_ids[valid]).float().mean() if valid.any() else rel_logits.sum() * 0.0

        return {
            "total_loss": losses["total_loss"],
            "future_mask_loss": losses.get("future_mask_loss", torch.tensor(0.0)),
            "goal_mask_loss": losses.get("goal_mask_loss", torch.tensor(0.0)),
            "relation_loss": losses.get("relation_loss", torch.tensor(0.0)),
            "future_dice": future_dice,
            "goal_dice": goal_dice,
            "relation_acc": rel_acc,
            "transition_tokens": transition_tokens,
        }

    @staticmethod
    def _dice(logits, target, eps=1e-6):
        pred = (torch.sigmoid(logits) > 0.5).float()
        if target.dim() == 2:
            target = target.unsqueeze(1)
        elif target.dim() == 3:
            target = target.unsqueeze(1)
        inter = (pred * target).sum()
        union = pred.sum() + target.sum()
        return (2.0 * inter + eps) / (union + eps)
