"""
Gated Transition Action Adapter.
=================================
Injects bottleneck transition tokens into the VLM action context
via a gated cross-attention residual.

    conditioned_vl_embs = vl_embs + tanh(gate) * CrossAttn(q=vl_embs, kv=proj_transition)

Gate initialized near zero → initial behavior ≈ original LaRA-VLA action_only.
"""

import torch
import torch.nn as nn


class TransitionToActionProjector(nn.Module):
    """Project bottleneck transition tokens [B, Kt, 512] → [B, Kt, 2560]."""

    def __init__(self, transition_dim: int = 512, num_tokens: int = 6, vlm_dim: int = 2560):
        super().__init__()
        self.project = nn.Linear(transition_dim, vlm_dim)

    def forward(self, transition_tokens):
        """[B, Kt, Dt] → [B, Kt, Dvlm]"""
        return self.project(transition_tokens)


class GatedTransitionActionAdapter(nn.Module):
    """
    Inject transition tokens into VLM action context via gated cross-attention.

    vl_embs:            [B, L, 2560]  — original VLM hidden states
    transition_tokens:  [B, Kt, 512]  — bottleneck transition tokens

    Returns conditioned_vl_embs: [B, L, 2560]
    """

    def __init__(
        self,
        transition_dim: int = 512,
        num_transition_tokens: int = 6,
        vlm_dim: int = 2560,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.projector = TransitionToActionProjector(
            transition_dim=transition_dim, num_tokens=num_transition_tokens, vlm_dim=vlm_dim
        )
        self.cross_attn = nn.MultiheadAttention(
            vlm_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(vlm_dim)
        # Gate: initialized near zero so initial behavior ≈ original action_only
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, vl_embs, transition_tokens):
        """
        Args:
            vl_embs:           [B, L, 2560]
            transition_tokens: [B, Kt, 512]

        Returns:
            conditioned_vl_embs: [B, L, 2560]
        """
        # Project transition tokens to VLM dim
        proj = self.projector(transition_tokens)  # [B, Kt, 2560]

        # Cross-attention: vl_embs attends to transition tokens
        attn_out, _ = self.cross_attn(
            query=vl_embs, key=proj, value=proj
        )
        attn_out = self.attn_norm(attn_out)

        # Gated residual
        gate_val = torch.tanh(self.gate)  # ≈ 0 initially
        return vl_embs + gate_val * attn_out
