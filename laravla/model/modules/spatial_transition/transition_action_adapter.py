"""
Gated Transition Action Adapter (Step 6A: + spatial stream).
==============================================================
Injects bottleneck transition tokens + predicted DINO subgoal +
proprioception into the VLM action context via gated cross-attention.

    conditioned_vl_embs = vl_embs + tanh(gate) * CrossAttn(q=vl_embs, kv=proj_all)

Gate initialized near zero → initial behavior ≈ original LaRA-VLA action_only.
"""

import torch
import torch.nn as nn


class TransitionToActionProjector(nn.Module):
    """Project tokens [B, Kt, Dt] → [B, Kt, Dvlm]. Kt is dynamic (6 or 8)."""

    def __init__(self, transition_dim: int = 512, num_tokens: int = 6, vlm_dim: int = 2560):
        super().__init__()
        self.project = nn.Linear(transition_dim, vlm_dim)

    def forward(self, transition_tokens):
        """[B, Kt, Dt] → [B, Kt, Dvlm]"""
        return self.project(transition_tokens)


class ProprioEncoder(nn.Module):
    """Encode proprioception [B, 7] → [B, 1, transition_dim]."""

    def __init__(self, state_dim: int = 7, transition_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, transition_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(transition_dim // 2, transition_dim),
            nn.LayerNorm(transition_dim),
        )

    def forward(self, state):
        """[B, D_state] → [B, 1, transition_dim]"""
        if state.dim() == 3:
            state = state.squeeze(1)  # [B, 1, D] → [B, D]
        return self.encoder(state.float()).unsqueeze(1)


class GatedTransitionActionAdapter(nn.Module):
    """
    Inject transition tokens + spatial tokens into VLM action context.

    vl_embs:            [B, L, 2560]  — original VLM hidden states
    transition_tokens:  [B, Kt, 512]  — transition + spatial tokens (6 or 8)

    Returns conditioned_vl_embs: [B, L, 2560]
    """

    def __init__(
        self,
        transition_dim: int = 512,
        num_transition_tokens: int = 8,  # 6 typed + 2 spatial (Step 6A)
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
        # Sigmoid gate: logit=-2.2 → sigmoid≈0.1 initially (spatial active but weak)
        self.gate_logit = nn.Parameter(torch.tensor(-2.2))

    def forward(self, vl_embs, transition_tokens):
        """
        Args:
            vl_embs:           [B, L, 2560]
            transition_tokens: [B, Kt, 512]  (6 typed + 2 spatial)

        Returns:
            conditioned_vl_embs: [B, L, 2560]
        """
        proj = self.projector(transition_tokens)  # [B, Kt, 2560]
        attn_out, _ = self.cross_attn(query=vl_embs, key=proj, value=proj)
        attn_out = self.attn_norm(attn_out)
        gate_val = torch.sigmoid(self.gate_logit)  # ∈(0,1), init≈0.1
        return vl_embs + gate_val * attn_out
