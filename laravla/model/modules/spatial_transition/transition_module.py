"""
MaskConditionedTransitionModule: bottleneck cross-attention transition.
========================================================================
Input:  vlm_projected [B, L, transition_dim]
        mask_tokens   [B, K, transition_dim]
Output: transition_tokens [B, Kt, transition_dim]

Lightweight: operates entirely in bottleneck space (default 512-dim).
"""

import torch
import torch.nn as nn


class MaskConditionedTransitionModule(nn.Module):
    """Learn latent transition tokens via cross-attention in bottleneck space."""

    def __init__(
        self,
        transition_dim: int = 512,
        num_transition_tokens: int = 6,
        num_heads: int = 4,
        ffn_multiplier: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.transition_dim = transition_dim
        self.num_transition_tokens = num_transition_tokens

        self.transition_queries = nn.Parameter(
            torch.randn(1, num_transition_tokens, transition_dim) * 0.02
        )
        # Per-token identity embedding — prevents all tokens from collapsing to
        # the same representation (token collapse fix, Step 5 pre-fix).
        self.token_identity = nn.Parameter(
            torch.randn(1, num_transition_tokens, transition_dim) * 0.02
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(
                    transition_dim, num_heads, dropout=dropout, batch_first=True
                ),
                "cross_norm": nn.LayerNorm(transition_dim),
                "self_attn": nn.MultiheadAttention(
                    transition_dim, num_heads, dropout=dropout, batch_first=True
                ),
                "self_norm": nn.LayerNorm(transition_dim),
                "ffn": nn.Sequential(
                    nn.Linear(transition_dim, transition_dim * ffn_multiplier),
                    nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(transition_dim * ffn_multiplier, transition_dim),
                    nn.Dropout(dropout),
                ),
                "ffn_norm": nn.LayerNorm(transition_dim),
            }))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, vlm_projected, mask_tokens=None):
        """
        [B,L,D] + optional [B,K,D] → [B,Kt,D]  where D=transition_dim.

        When mask_tokens is None (no-mask mode), transition queries attend
        to VLM projected tokens only. This is the mask-supervised but
        mask-free-inference path.
        """
        B = vlm_projected.shape[0]
        if mask_tokens is not None:
            context = torch.cat([vlm_projected, mask_tokens], dim=1)
        else:
            context = vlm_projected
        # Add per-token identity to prevent collapse (all tokens becoming identical)
        queries = self.transition_queries.expand(B, -1, -1) + self.token_identity.expand(B, -1, -1)

        for layer in self.layers:
            cross_out, _ = layer["cross_attn"](query=queries, key=context, value=context)
            queries = layer["cross_norm"](queries + cross_out)
            self_out, _ = layer["self_attn"](query=queries, key=queries, value=queries)
            queries = layer["self_norm"](queries + self_out)
            queries = layer["ffn_norm"](queries + layer["ffn"](queries))

        return queries
