"""
Unified DINO future loss and metric.
=====================================
Single implementation used by: P1 train, P1 eval, P1 round-trip,
P2 auxiliary monitoring, P2 parity, independent validation.

No other DINO cosine/loss implementation should exist in the project.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def dino_future_cosine(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
) -> dict:
    """
    Compute DINO future cosine similarity and loss.

    Args:
        pred:   [B, K, D] predicted future DINO patch features
        target: [B, K, D] frozen DINO encoder output
        valid:  [B] bool tensor (True = use this sample).
                If None, all samples are treated as valid.

    Returns:
        dict with:
            "loss":         scalar, 1 - mean_cosine  (0 = perfect)
            "cosine":       scalar or None, mean cosine over valid samples
            "valid_count":  int, number of valid samples
    """
    eps = 1e-6
    pred = F.normalize(pred.float(), dim=-1, eps=eps)
    target = F.normalize(target.float(), dim=-1, eps=eps)

    # Per-patch cosine → per-sample mean → [B]
    patch_cos = (pred * target).sum(dim=-1)  # [B, K]
    sample_cos = patch_cos.mean(dim=-1)      # [B]

    if valid is None:
        valid = torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)
    else:
        valid = valid.bool().to(pred.device)

    valid_count = int(valid.sum().item())

    if valid_count == 0:
        zero = pred.sum() * 0.0
        return {"loss": zero, "cosine": None, "valid_count": 0}

    cosine = sample_cos[valid].mean()
    loss = 1.0 - cosine

    return {
        "loss": loss,
        "cosine": cosine.detach(),
        "valid_count": valid_count,
    }
