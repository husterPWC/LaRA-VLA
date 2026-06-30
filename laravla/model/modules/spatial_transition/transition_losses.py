"""
Transition losses: future mask, goal mask, relation classification.
====================================================================
- mask_loss = BCEWithLogits + Dice
- relation_loss = CrossEntropy
- transition_total_loss = weighted sum
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred_logits, target, eps=1e-6):
    """
    Soft Dice loss for binary segmentation.

    Args:
        pred_logits: [B, 1, H, W] logits
        target:      [B, H, W] or [B, 1, H, W] binary mask
        eps:         smoothing epsilon

    Returns:
        scalar loss
    """
    pred = torch.sigmoid(pred_logits)
    if target.dim() == 3:
        target = target.unsqueeze(1)  # [B, H, W] → [B, 1, H, W]
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return (1.0 - dice).mean()


def mask_loss(pred_logits, target, bce_weight=1.0, dice_weight=1.0):
    """
    Combined BCE + Dice mask loss.

    Args:
        pred_logits: [B, 1, H, W]
        target:      [B, H, W] float32 binary mask
        bce_weight:  BCE loss weight
        dice_weight: Dice loss weight

    Returns:
        scalar loss
    """
    if target.dim() == 3:
        target = target.unsqueeze(1)  # [B, 1, H, W]

    bce = F.binary_cross_entropy_with_logits(pred_logits, target)
    dice = dice_loss(pred_logits, target)
    return bce_weight * bce + dice_weight * dice


def relation_loss(pred_logits, target_ids, ignore_index=-1):
    """
    Cross-entropy loss for relation classification.

    Args:
        pred_logits: [B, num_classes]
        target_ids:  [B] LongTensor with class indices

    Returns:
        scalar loss
    """
    # Filter out invalid labels (e.g., -1 or out of range)
    valid_mask = (target_ids >= 0) & (target_ids < pred_logits.shape[1])
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred_logits.device, requires_grad=True)

    return F.cross_entropy(
        pred_logits[valid_mask],
        target_ids[valid_mask].long(),
    )


def transition_total_loss(
    future_logits=None,
    future_target=None,
    goal_logits=None,
    goal_target=None,
    relation_logits=None,
    relation_target=None,
    w_future=0.05,
    w_goal=0.10,
    w_relation=0.05,
):
    """
    Compute total transition loss.

    Args:
        future_logits:    [B, 1, R, R] or None
        future_target:    [B, H, W] float32 or None
        goal_logits:      [B, 1, R, R] or None
        goal_target:      [B, H, W] float32 or None
        relation_logits:  [B, C] or None
        relation_target:  [B] LongTensor or None
        w_future, w_goal, w_relation: loss weights

    Returns:
        dict with individual losses and total_loss
    """
    losses = {}
    total = torch.tensor(0.0, device=future_logits.device if future_logits is not None else "cpu")

    if future_logits is not None and future_target is not None:
        L_future = mask_loss(future_logits, future_target)
        losses["future_mask_loss"] = L_future
        total = total + w_future * L_future

    if goal_logits is not None and goal_target is not None:
        L_goal = mask_loss(goal_logits, goal_target)
        losses["goal_mask_loss"] = L_goal
        total = total + w_goal * L_goal

    if relation_logits is not None and relation_target is not None:
        L_rel = relation_loss(relation_logits, relation_target)
        losses["relation_loss"] = L_rel
        total = total + w_relation * L_rel

    losses["total_loss"] = total
    return losses
