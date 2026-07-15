"""
P1TransitionWrapper: lightweight wrapper for DDP-safe P1 training.
===================================================================
Only wraps the 6 P1 trainable modules (~26.5M). Qwen-VL and Action
model stay outside DDP — each rank loads its own frozen copy locally.

P1NoMaskWrapper: mask-supervised but mask-free-inference variant.
===================================================================
No mask_token_encoder — transition tokens are learned from VLM hidden
only. Adds current_mask_decoder so the model learns to ground current
objects from RGB alone. current/future/goal masks are supervision only.

Usage in training script:
    vla = build_framework(cfg)
    vla.load_state_dict(...)
    # Freeze VLA
    for p in vla.parameters(): p.requires_grad_(False)
    # Build P1 wrapper (trainable)
    p1_model = P1TransitionWrapper(vla).to(device)       # old: mask-conditioned
    p1_model = P1NoMaskWrapper(vla).to(device)           # new: RGB-only
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


class P1NoMaskWrapper(nn.Module):
    """
    Mask-supervised, mask-free-inference P1 wrapper.
    ================================================
    Key differences from P1TransitionWrapper:
      - No mask_token_encoder (current_mask is NOT an input).
      - Adds current_mask_decoder — the model must predict current mask
        from RGB alone, learning implicit spatial grounding.
      - transition_module called WITHOUT mask_tokens (context = VLM only).

    Trainable modules (all in bottleneck 512-dim):
      - vlm_projector         2560 → 512
      - transition_module     cross-attention (VLM only, no mask context)
      - current_mask_decoder  predict current mask from transition tokens
      - future_mask_decoder   predict future mask
      - goal_mask_decoder     predict goal mask
      - relation_head         6-class relation classifier
    """

    def __init__(self, vla):
        super().__init__()
        self.vlm_projector = vla.vlm_projector
        self.transition_module = vla.transition_module
        self.current_mask_decoder = vla.current_mask_decoder
        self.future_mask_decoder = vla.future_mask_decoder
        self.goal_mask_decoder = vla.goal_mask_decoder
        self.relation_head = vla.relation_head

        self.loss_weights = vla.transition_loss_weights
        self.mask_res = 56

    def forward(self, vlm_hidden, cur_masks, future_masks, goal_masks, rel_ids):
        """
        Args:
            vlm_hidden:    [B, L, 2560] frozen Qwen-VL hidden states (RGB + instruction only)
            cur_masks:     [B, 1, H, W] current mask SUPERVISION (not input!)
            future_masks:  [B, H, W]    future mask GT
            goal_masks:    [B, H, W]    goal mask GT
            rel_ids:       [B] LongTensor relation label ids

        Returns:
            dict with total_loss, current/future/goal_mask_loss, relation_loss,
                 current/future/goal_dice, relation_acc, transition_tokens
        """
        from laravla.model.modules.spatial_transition import transition_total_loss

        # Bottleneck projection: VLM hidden → 512-dim
        vlm_proj = self.vlm_projector(vlm_hidden.float())

        # Transition: NO mask tokens — cross-attend to VLM only
        # This forces the model to learn spatial grounding from RGB alone.
        transition_tokens = self.transition_module(vlm_proj, mask_tokens=None)

        # Decode all masks from transition tokens
        current_logits = self.current_mask_decoder(transition_tokens)
        future_logits = self.future_mask_decoder(transition_tokens)
        goal_logits = self.goal_mask_decoder(transition_tokens)
        rel_logits = self.relation_head(transition_tokens)

        # Resize GT masks to match decoder output resolution
        R = future_logits.shape[-1]
        cur_gt = F.interpolate(
            cur_masks.float(), size=(R, R), mode='nearest'
        ).squeeze(1)
        future_gt = F.interpolate(
            future_masks.unsqueeze(1), size=(R, R), mode='nearest'
        ).squeeze(1)
        goal_gt = F.interpolate(
            goal_masks.unsqueeze(1), size=(R, R), mode='nearest'
        ).squeeze(1)

        # Losses: current + future + goal + relation
        w = self.loss_weights
        losses = transition_total_loss(
            current_logits=current_logits, current_target=cur_gt,
            future_logits=future_logits, future_target=future_gt,
            goal_logits=goal_logits, goal_target=goal_gt,
            relation_logits=rel_logits, relation_target=rel_ids,
            w_current=w.get("current_mask", 0.05),
            w_future=w.get("future_mask", 0.05),
            w_goal=w.get("goal_mask", 0.10),
            w_relation=w.get("relation", 0.05),
        )

        # Metrics
        cur_dice = self._dice(current_logits, cur_gt)
        future_dice = self._dice(future_logits, future_gt)
        goal_dice = self._dice(goal_logits, goal_gt)
        rel_pred = rel_logits.argmax(dim=1)
        valid = (rel_ids >= 0) & (rel_ids < rel_logits.shape[1])
        rel_acc = (rel_pred[valid] == rel_ids[valid]).float().mean() if valid.any() else rel_logits.sum() * 0.0

        return {
            "total_loss": losses["total_loss"],
            "current_mask_loss": losses.get("current_mask_loss", torch.tensor(0.0)),
            "future_mask_loss": losses.get("future_mask_loss", torch.tensor(0.0)),
            "goal_mask_loss": losses.get("goal_mask_loss", torch.tensor(0.0)),
            "relation_loss": losses.get("relation_loss", torch.tensor(0.0)),
            "current_dice": cur_dice,
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
