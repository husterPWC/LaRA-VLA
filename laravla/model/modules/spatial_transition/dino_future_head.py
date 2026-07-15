"""
DINOFutureHead — predict future DINO patch features from transition tokens.
=============================================================================
256 learnable patch queries cross-attend to transition tokens and decode
a full spatial future feature map [B, 256, dino_dim] (e.g. 768 for ViT-B/14).

Used in P1: auxiliary supervision — transition tokens must learn to predict
what the scene will look like in DINO feature space after the current subtask
action completes.

Reference: LaWAM's LAMDecoder_v2 predicts future DINO features from latent
action codes using AdaLN blocks. We use a simpler query-based design.
"""

import torch
import torch.nn as nn


class DINOFutureHead(nn.Module):
    """
    Predict future DINO patch features from transition tokens.

    Architecture:
        transition_tokens [B, T, 512]
            ↓
        256 learnable patch queries cross-attend to transition tokens
            ↓
        LayerNorm → Linear(512→512) → GELU → Linear(512→dino_dim)
            ↓
        pred_future_dino [B, 256, dino_dim]

    Args:
        transition_dim: Transition token dimension (default 512)
        dino_dim: DINO feature dimension (768 for ViT-B/14, 384 for ViT-S/14)
        num_patches: Number of DINO patches (256 for 224×224 with patch_size=14)
        num_heads: Cross-attention heads
    """

    def __init__(
        self,
        transition_dim: int = 512,
        dino_dim: int = 768,
        num_patches: int = 256,
        num_heads: int = 8,
    ):
        super().__init__()
        self.transition_dim = transition_dim
        self.dino_dim = dino_dim
        self.num_patches = num_patches

        # Learnable patch queries — one per DINO patch position
        self.patch_queries = nn.Parameter(
            torch.randn(1, num_patches, transition_dim) * 0.02
        )

        # Cross-attention: patch queries attend to transition tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=transition_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(transition_dim)

        # Output projection: transition_dim → dino_dim
        self.out = nn.Sequential(
            nn.Linear(transition_dim, transition_dim),
            nn.GELU(),
            nn.Linear(transition_dim, dino_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, transition_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            transition_tokens: [B, T, transition_dim]  (T=6 normally)

        Returns:
            pred_future_dino: [B, num_patches, dino_dim]
        """
        B = transition_tokens.shape[0]
        q = self.patch_queries.expand(B, -1, -1)  # [B, 256, 512]

        # Cross-attend from patch queries to transition tokens
        x, _ = self.cross_attn(query=q, key=transition_tokens, value=transition_tokens)
        x = self.norm(q + x)  # residual + norm
        x = self.out(x)       # [B, 256, dino_dim]

        return x


def dino_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity loss for DINO feature prediction.

    Args:
        pred:   [B, K, D] predicted future DINO features
        target: [B, K, D] frozen DINO encoder output (stopgrad'd by caller)

    Returns:
        scalar loss = 1 - mean(cosine_sim(pred, target))
    """
    pred_n = torch.nn.functional.normalize(pred.float(), dim=-1)
    target_n = torch.nn.functional.normalize(target.float(), dim=-1)
    cos_sim = (pred_n * target_n).sum(dim=-1)  # [B, K]
    return 1.0 - cos_sim.mean()


def dino_cosine_similarity(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean cosine similarity for logging (higher is better).

    Args:
        pred:   [B, K, D]
        target: [B, K, D]

    Returns:
        scalar in [-1, 1]
    """
    pred_n = torch.nn.functional.normalize(pred.float(), dim=-1)
    target_n = torch.nn.functional.normalize(target.float(), dim=-1)
    return (pred_n * target_n).sum(dim=-1).mean()
