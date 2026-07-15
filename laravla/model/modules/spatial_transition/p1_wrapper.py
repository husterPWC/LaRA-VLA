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
    Mask-supervised, mask-free-inference P1 wrapper (v3: shared decoder).
    ======================================================================
    - No mask_token_encoder. Transition from VLM hidden only.
    - SHARED mask decoder for current/future/goal — forces token differentiation.
    - Type embedding at BOTH query input and output.
    - Inter-type diversity loss (w=0.05) prevents cross-type collapse.

    Trainable modules:
      - vlm_projector, transition_module, mask_decoder (shared),
        relation_head, dino_future_head
    """

    def __init__(self, vla):
        super().__init__()
        self.vlm_projector = vla.vlm_projector
        self.transition_module = vla.transition_module
        # Shared mask decoder: current/future/goal all use the SAME decoder.
        # If tokens collapse → same mask for all targets → loss forces differentiation.
        self.mask_decoder = vla.current_mask_decoder  # shared!
        self.relation_head = vla.relation_head
        self.dino_future_head = getattr(vla, 'dino_future_head', None)

        self.loss_weights = vla.transition_loss_weights
        self.mask_res = 56

        # Type embedding (same param as transition_module for output reinforcement)
        self.type_embedding = self.transition_module.type_embedding

    def forward(self, vlm_hidden, cur_masks, future_masks, goal_masks, rel_ids,
                dino_future_target=None, tau_future_valid=None):
        """
        Args:
            vlm_hidden:         [B, L, 2560] frozen Qwen-VL hidden states
            cur_masks:          [B, 1, H, W] current mask SUPERVISION (not input!)
            future_masks:       [B, H, W]    future mask GT (tau future or original)
            goal_masks:         [B, H, W]    goal mask GT
            rel_ids:            [B] LongTensor relation label ids
            dino_future_target: [B, 256, dino_dim] or None
            tau_future_valid:   [B] bool tensor or None. If provided, masks
                                future_mask_loss and dino_future_loss for
                                samples where tau gap is too small.

        Returns:
            dict with total_loss, individual losses, metrics, transition_tokens
        """
        from laravla.model.modules.spatial_transition import (
            transition_total_loss, dino_cosine_loss, dino_cosine_similarity,
            token_diversity_loss
        )

        # Bottleneck projection: VLM hidden → 512-dim
        vlm_proj = self.vlm_projector(vlm_hidden.float())

        # Transition: NO mask tokens — cross-attend to VLM only
        # Slot identity residual (v4): the transition module's self-attention
        # homogenizes token queries (q_init inter=0.001 → z_raw inter=0.88).
        # We add the ORIGINAL diverse queries as a strong residual so token
        # identity survives through the module.
        B = vlm_proj.shape[0]
        q_init = (self.transition_module.transition_queries.expand(B, -1, -1)
                  + self.transition_module.type_embedding.expand(B, -1, -1))
        transition_tokens = self.transition_module(vlm_proj, mask_tokens=None)

        # Strong residual: LayerNorm'd original diverse queries survive.
        # LN(q_init) normalizes each query to ~unit norm so the residual
        # contribution is scale-invariant regardless of absolute norm.
        GAMMA = self.loss_weights.get("slot_residual_gamma", 2.0)
        q_init_ln = F.layer_norm(q_init.float(), [q_init.shape[-1]])
        z_typed = transition_tokens + GAMMA * q_init_ln
        z_typed = F.layer_norm(z_typed.float(), [z_typed.shape[-1]])
        # transition_tokens: [B, 6, 512]
        #   [0,1] = current type  [2,3] = future type
        #   [4]   = goal type     [5]   = relation type

        current_tokens  = z_typed[:, 0:2, :]   # [B, 2, 512]
        future_tokens   = z_typed[:, 2:4, :]   # [B, 2, 512]
        goal_tokens     = z_typed[:, 4:5, :]   # [B, 1, 512]
        relation_tokens = z_typed[:, 5:6, :]   # [B, 1, 512]

        # Shared mask decoder: current/future/goal all use the SAME decoder.
        # If tokens collapse → shared decoder predicts same mask for all →
        # loss penalizes because targets differ → gradient forces tokens apart.
        current_logits = self.mask_decoder(current_tokens)
        future_logits  = self.mask_decoder(future_tokens)
        goal_logits    = self.mask_decoder(goal_tokens)
        rel_logits     = self.relation_head(relation_tokens)

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

        # ── Spatial losses: current + future + goal + relation ──
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

        # ── Loss masking for tau_future_valid (Step 4) ────────────
        # When tau_future_valid=False, future gap is too small — mask future
        # and DINO losses (but keep current/goal/relation intact).
        if tau_future_valid is not None:
            valid_mask = tau_future_valid.float().to(losses["total_loss"].device)
            valid_ratio = valid_mask.mean()
            if valid_ratio < 1.0 and valid_ratio > 0:
                # Scale future mask loss by valid ratio
                if "future_mask_loss" in losses:
                    losses["future_mask_loss"] = losses["future_mask_loss"] * valid_ratio
                losses["total_loss"] = losses["total_loss"] - (
                    w.get("future_mask", 0.05) *
                    losses.get("future_mask_loss", torch.tensor(0.0)) / max(valid_ratio, 0.01) * (1 - valid_ratio)
                )
        else:
            valid_ratio = torch.tensor(1.0)

        # ── Token diversity loss (anti-collapse v3) ──────────
        # Only penalize INTER-type cosine (current vs future, current vs goal, etc.)
        # Intra-type similarity is acceptable (e.g., two current tokens can be similar).
        inter_type_cos = _compute_inter_type_cos(transition_tokens)  # see helper below
        excess = F.relu(inter_type_cos - 0.5)
        L_div = 0.05 * (excess ** 2).mean()  # stronger weight (was 0.01)

        total = losses["total_loss"] + L_div
        result = {
            "token_diversity_loss": L_div,
            "current_mask_loss": losses.get("current_mask_loss", torch.tensor(0.0)),
            "future_mask_loss": losses.get("future_mask_loss", torch.tensor(0.0)),
            "goal_mask_loss": losses.get("goal_mask_loss", torch.tensor(0.0)),
            "relation_loss": losses.get("relation_loss", torch.tensor(0.0)),
            "tau_valid_ratio": valid_ratio.detach() if isinstance(valid_ratio, torch.Tensor) else torch.tensor(valid_ratio),
        }

        # ── DINO future loss (Step 3) ────────────────────────────
        if self.dino_future_head is not None and dino_future_target is not None:
            pred_dino = self.dino_future_head(future_tokens)              # [B, 256, dino_dim] (reads future type)
            L_dino = dino_cosine_loss(pred_dino, dino_future_target)      # scalar (averaged over batch)
            cos_dino = dino_cosine_similarity(pred_dino, dino_future_target)

            # Mask DINO loss for invalid tau samples
            if tau_future_valid is not None:
                dino_valid_mask = tau_future_valid.float().to(L_dino.device)
                dino_valid_ratio = dino_valid_mask.mean()
                if dino_valid_ratio < 1.0 and dino_valid_ratio > 0:
                    # Recompute per-sample DINO loss with masking
                    pred_n = torch.nn.functional.normalize(pred_dino.float(), dim=-1)
                    target_n = torch.nn.functional.normalize(dino_future_target.float(), dim=-1)
                    per_sample_loss = 1.0 - (pred_n * target_n).sum(dim=-1).mean(dim=-1)  # [B]
                    L_dino_masked = (per_sample_loss * dino_valid_mask).sum() / max(dino_valid_mask.sum(), 1)
                    L_dino = L_dino_masked
                    result["tau_valid_ratio"] = dino_valid_ratio.detach()

            w_dino = w.get("dino_future", 0.05)
            total = total + w_dino * L_dino

            result["dino_future_loss"] = L_dino
            result["dino_future_cos"] = cos_dino
        else:
            L_dino = None

        # ── Metrics ──────────────────────────────────────────────
        cur_dice = self._dice(current_logits, cur_gt)
        future_dice = self._dice(future_logits, future_gt)
        goal_dice = self._dice(goal_logits, goal_gt)
        rel_pred = rel_logits.argmax(dim=1)
        valid = (rel_ids >= 0) & (rel_ids < rel_logits.shape[1])
        rel_acc = (rel_pred[valid] == rel_ids[valid]).float().mean() if valid.any() else rel_logits.sum() * 0.0

        # ── Latent diagnostics (collapse detection) ───────────
        # Inter-type vs intra-type pair cosine
        inter_cos, intra_cos = _compute_inter_intra_cos(transition_tokens)
        latent_var = transition_tokens.var(dim=0).mean()
        token_norms = transition_tokens.float().norm(dim=-1)
        latent_norm_mean = token_norms.mean()
        latent_norm_std = token_norms.std()

        result.update({
            "total_loss": total,
            "current_dice": cur_dice,
            "future_dice": future_dice,
            "goal_dice": goal_dice,
            "relation_acc": rel_acc,
            "latent_var": latent_var,
            "latent_pair_cos": inter_cos,       # inter-type (key metric)
            "intra_pair_cos": intra_cos,         # intra-type (acceptable if higher)
            "latent_norm_mean": latent_norm_mean,
            "latent_norm_std": latent_norm_std,
            "transition_tokens": transition_tokens,
        })
        return result

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


# ── Type-aware helpers (module-level) ──────────────────────────

def _compute_inter_intra_cos(z):
    import torch.nn.functional as F
    z_n = F.normalize(z.float(), dim=-1)
    z_mean = z_n.mean(dim=0)  # [6, D]
    sim = z_mean @ z_mean.T  # [6, 6]
    inter_vals, intra_vals = [], []
    for i in range(6):
        ti = _token_type(i)
        for j in range(i + 1, 6):
            tj = _token_type(j)
            if ti == tj:
                intra_vals.append(sim[i, j])
            else:
                inter_vals.append(sim[i, j])
    inter_cos = torch.stack(inter_vals).mean() if inter_vals else torch.tensor(0.0)
    intra_cos = torch.stack(intra_vals).mean() if intra_vals else torch.tensor(0.0)
    return inter_cos, intra_cos


def _compute_inter_type_cos(z):
    inter_cos, _ = _compute_inter_intra_cos(z)
    return inter_cos


def _token_type(idx):
    if idx < 2: return 0
    elif idx < 4: return 1
    elif idx < 5: return 2
    else: return 3
