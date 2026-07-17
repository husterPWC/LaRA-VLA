"""
P1TransitionWrapper + P1NoMaskWrapper — DDP-safe P1 training.
===============================================================
P1TransitionWrapper: legacy mask-conditioned (kept for ablation).
P1NoMaskWrapper:    formal unified — uses SpatialTransitionBackbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class P1TransitionWrapper(nn.Module):
    """Legacy mask-conditioned P1 wrapper (kept for ablation only)."""

    def __init__(self, vla):
        super().__init__()
        self.vlm_projector = vla.vlm_projector
        self.mask_token_encoder = vla.mask_token_encoder
        self.transition_module = vla.transition_module
        self.future_mask_decoder = vla.future_mask_decoder
        self.goal_mask_decoder = vla.goal_mask_decoder
        self.relation_head = vla.relation_head
        self.loss_weights = vla.transition_loss_weights
        self.mask_res = 56

    def forward(self, vlm_hidden, cur_masks, future_masks, goal_masks, rel_ids):
        from laravla.model.modules.spatial_transition import transition_total_loss
        vlm_proj = self.vlm_projector(vlm_hidden.float())
        mask_tokens = self.mask_token_encoder(cur_masks)
        transition_tokens = self.transition_module(vlm_proj, mask_tokens)
        future_logits = self.future_mask_decoder(transition_tokens)
        goal_logits = self.goal_mask_decoder(transition_tokens)
        rel_logits = self.relation_head(transition_tokens)
        R = future_logits.shape[-1]
        future_gt = F.interpolate(future_masks.unsqueeze(1), size=(R,R), mode='nearest').squeeze(1)
        goal_gt = F.interpolate(goal_masks.unsqueeze(1), size=(R,R), mode='nearest').squeeze(1)
        w = self.loss_weights
        losses = transition_total_loss(
            future_logits=future_logits, future_target=future_gt,
            goal_logits=goal_logits, goal_target=goal_gt,
            relation_logits=rel_logits, relation_target=rel_ids,
            w_future=w.get("future_mask",0.05), w_goal=w.get("goal_mask",0.10),
            w_relation=w.get("relation",0.05))
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
            "future_dice": future_dice, "goal_dice": goal_dice,
            "relation_acc": rel_acc, "transition_tokens": transition_tokens,
        }

    @staticmethod
    def _dice(logits, target, eps=1e-6):
        pred = (torch.sigmoid(logits) > 0.5).float()
        if target.dim() == 2: target = target.unsqueeze(1)
        elif target.dim() == 3: target = target.unsqueeze(1)
        inter = (pred * target).sum()
        union = pred.sum() + target.sum()
        return (2.0 * inter + eps) / (union + eps)


class P1NoMaskWrapper(nn.Module):
    """
    Formal unified P1 wrapper using SpatialTransitionBackbone.
    ===========================================================
    Student: backbone(vlm_hidden) → masks, relation, DINO prediction.
    Teacher: posterior_encoder (optional, training-only).

    Checkpoint format:
        {"p1_state_dict": wrapper.state_dict()}
        → backbone params stored as "backbone.*" keys.
    """

    def __init__(self, backbone, loss_weights=None, mask_res=56):
        super().__init__()
        self.backbone = backbone
        self.loss_weights = loss_weights or {}
        self.mask_res = mask_res
        self.register_buffer('_distill_step', torch.tensor(0, dtype=torch.long))

    def forward(self, vlm_hidden, cur_masks, future_masks, goal_masks, rel_ids,
                tau_future_valid=None):
        """
        Returns dict with total_loss, individual losses, metrics, transition_tokens.
        """
        from laravla.model.modules.spatial_transition import (
            transition_total_loss, token_diversity_loss
        )
        w = self.loss_weights

        # ── Student forward ──────────────────────────────────
        out = self.backbone(vlm_hidden)

        # Resize GT masks
        R = out.future_mask_logits.shape[-1]
        cur_gt = F.interpolate(cur_masks.float(), size=(R,R), mode='nearest').squeeze(1)
        future_gt = F.interpolate(future_masks.unsqueeze(1), size=(R,R), mode='nearest').squeeze(1)
        goal_gt = F.interpolate(goal_masks.unsqueeze(1), size=(R,R), mode='nearest').squeeze(1)

        # Spatial losses
        losses = transition_total_loss(
            current_logits=out.current_mask_logits, current_target=cur_gt,
            future_logits=out.future_mask_logits, future_target=future_gt,
            goal_logits=out.goal_mask_logits, goal_target=goal_gt,
            relation_logits=out.relation_logits, relation_target=rel_ids,
            w_current=w.get("current_mask",0.05), w_future=w.get("future_mask",0.05),
            w_goal=w.get("goal_mask",0.10), w_relation=w.get("relation",0.05),
        )

        # Token diversity
        L_div = token_diversity_loss(out.z_student, weight=0.01, threshold=0.5)
        total = losses["total_loss"] + L_div

        result = {
            "current_mask_loss": losses.get("current_mask_loss", torch.tensor(0.0)),
            "future_mask_loss": losses.get("future_mask_loss", torch.tensor(0.0)),
            "goal_mask_loss": losses.get("goal_mask_loss", torch.tensor(0.0)),
            "relation_loss": losses.get("relation_loss", torch.tensor(0.0)),
        }
        if tau_future_valid is not None:
            result["tau_valid_ratio"] = tau_future_valid.float().mean().detach()

        # ── Metrics ──────────────────────────────────────────
        cur_dice = self._dice(out.current_mask_logits, cur_gt)
        future_dice = self._dice(out.future_mask_logits, future_gt)
        goal_dice = self._dice(out.goal_mask_logits, goal_gt)
        rel_pred = out.relation_logits.argmax(dim=1)
        valid = (rel_ids >= 0) & (rel_ids < out.relation_logits.shape[1])
        rel_acc = (rel_pred[valid] == rel_ids[valid]).float().mean() if valid.any() else out.relation_logits.sum()*0.0
        z_n = F.normalize(out.z_student.float(), dim=-1).mean(dim=0)
        sim = z_n @ z_n.T
        mask = ~torch.eye(6, dtype=torch.bool, device=sim.device)
        latent_pair_cos = sim[mask].mean()

        result.update({
            "total_loss": total,
            "current_dice": cur_dice, "future_dice": future_dice,
            "goal_dice": goal_dice, "relation_acc": rel_acc,
            "latent_var": out.z_student.var(dim=0).mean(),
            "latent_pair_cos": latent_pair_cos,
            "latent_norm_mean": out.z_student.float().norm(dim=-1).mean(),
            "latent_norm_std": out.z_student.float().norm(dim=-1).std(),
            "transition_tokens": out.z_student,
        })
        return result

    @staticmethod
    def _dice(logits, target, eps=1e-6):
        pred = (torch.sigmoid(logits) > 0.5).float()
        if target.dim() == 2: target = target.unsqueeze(1)
        elif target.dim() == 3: target = target.unsqueeze(1)
        inter = (pred * target).sum()
        union = pred.sum() + target.sum()
        return (2.0 * inter + eps) / (union + eps)


# ── Type-aware helpers ──────────────────────────────────────

def _compute_inter_intra_cos(z):
    import torch.nn.functional as F
    z_n = F.normalize(z.float(), dim=-1).mean(dim=0); s = z_n @ z_n.T
    inter, intra = [], []
    for i in range(6):
        ti = 0 if i<2 else (1 if i<4 else (2 if i<5 else 3))
        for j in range(i+1,6):
            tj = 0 if j<2 else (1 if j<4 else (2 if j<5 else 3))
            if ti==tj: intra.append(s[i,j])
            else: inter.append(s[i,j])
    ic = torch.stack(inter).mean() if inter else torch.tensor(0.0)
    ia = torch.stack(intra).mean() if intra else torch.tensor(0.0)
    return ic, ia

def _compute_inter_type_cos(z):
    ic, _ = _compute_inter_intra_cos(z)
    return ic

def _token_type(idx):
    if idx < 2: return 0
    elif idx < 4: return 1
    elif idx < 5: return 2
    else: return 3
