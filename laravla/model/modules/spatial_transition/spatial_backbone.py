"""
SpatialTransitionBackbone — single shared spatial reasoning module.
===================================================================
P1 and P2 MUST use this exact same backbone. No separate implementations.

Architecture:
    RGB + instruction → frozen VLM → vlm_hidden [B,L,2560]
    vlm_hidden → vlm_projector → [B,L,512]
    → transition_module (no mask tokens) → z_raw [B,6,512]
    → z_raw + gamma * LN(slot_queries) → z_student [B,6,512]
    → typed routing → shared mask_decoder / relation_head / dino_future_head

The backbone owns:
    vlm_projector, transition_module, slot queries, shared mask_decoder,
    relation_head, dino_future_head

Checkpoint format:
    {"spatial_backbone_state_dict": backbone.state_dict()}
    Load with strict=True → missing=0, unexpected=0.
"""

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpatialTransitionOutput:
    """All outputs from a single backbone forward pass."""
    z_student: torch.Tensor              # [B, 6, hidden_dim]

    current_tokens: torch.Tensor         # [B, 2, hidden_dim]
    future_tokens: torch.Tensor          # [B, 2, hidden_dim]
    goal_tokens: torch.Tensor            # [B, 1, hidden_dim]
    relation_tokens: torch.Tensor        # [B, 1, hidden_dim]

    current_mask_logits: torch.Tensor    # [B, 1, R, R]
    future_mask_logits: torch.Tensor     # [B, 1, R, R]
    goal_mask_logits: torch.Tensor       # [B, 1, R, R]
    relation_logits: torch.Tensor        # [B, num_classes]

    pred_future_dino: Optional[torch.Tensor] = None  # [B, K, dino_dim] or None


class SpatialTransitionBackbone(nn.Module):
    """
    Shared spatial reasoning backbone for P1 and P2.

    P1 uses this for mask/relation/DINO supervision.
    P2 uses this for z_student + pred_future_dino → action.

    Args:
        vlm_dim: VLM hidden dimension (2560 for Qwen3-VL-4B)
        hidden_dim: bottleneck dimension (default 512)
        num_slots: number of transition tokens (default 6)
        num_relation_labels: relation classification classes (default 7)
        gamma: slot identity residual strength (default 1.5)
        mask_res: mask decoder output resolution (default 56)
        dino_dim: DINO feature dimension (768 for ViT-B/14)
        dino_num_patches: DINO patch count (256 for 224x224/patch14)
    """

    def __init__(
        self,
        vlm_dim: int = 2560,
        hidden_dim: int = 512,
        num_slots: int = 6,
        num_relation_labels: int = 7,
        gamma: float = 1.5,
        mask_res: int = 56,
        dino_dim: int = 768,
        dino_num_patches: int = 256,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.gamma = gamma

        # ── Projector ─────────────────────────────────────────
        from laravla.model.modules.spatial_transition.mask_token_encoder import VLMProjector
        self.vlm_projector = VLMProjector(vlm_dim=vlm_dim, transition_dim=hidden_dim)

        # ── Transition module ─────────────────────────────────
        from laravla.model.modules.spatial_transition.transition_module import (
            MaskConditionedTransitionModule
        )
        self.transition_module = MaskConditionedTransitionModule(
            transition_dim=hidden_dim, num_transition_tokens=num_slots
        )

        # ── Slot queries (typed) ─────────────────────────────
        self.slot_queries = nn.Parameter(
            torch.randn(1, num_slots, hidden_dim) * 0.02
        )
        self.slot_norm = nn.LayerNorm(hidden_dim)

        # ── Shared mask decoder ───────────────────────────────
        from laravla.model.modules.spatial_transition.transition_decoders import (
            MaskDecoder, RelationHead
        )
        self.mask_decoder = MaskDecoder(
            transition_dim=hidden_dim, output_res=mask_res
        )

        # ── Relation head ─────────────────────────────────────
        self.relation_head = RelationHead(
            transition_dim=hidden_dim, num_classes=num_relation_labels
        )

        # ── DINO future head ──────────────────────────────────
        from laravla.model.modules.spatial_transition.dino_future_head import (
            DINOFutureHead
        )
        self.dino_future_head = DINOFutureHead(
            transition_dim=hidden_dim, dino_dim=dino_dim,
            num_patches=dino_num_patches,
        )

    # ── Public API ────────────────────────────────────────────

    def encode_student(self, vlm_hidden: torch.Tensor) -> torch.Tensor:
        """
        Encode VLM hidden → typed transition tokens.

        Args:
            vlm_hidden: [B, L, vlm_dim]

        Returns:
            z_student: [B, num_slots, hidden_dim]
        """
        context = self.vlm_projector(vlm_hidden.float())
        B = context.shape[0]

        q_init = self.slot_queries.expand(B, -1, -1)

        z_raw = self.transition_module(context, mask_tokens=None)

        # Slot identity residual + output normalization
        z_student = z_raw + self.gamma * self.slot_norm(q_init)
        z_student = F.layer_norm(z_student.float(), [z_student.shape[-1]])

        return z_student

    def decode_student(self, z_student: torch.Tensor) -> SpatialTransitionOutput:
        """
        Route typed tokens → mask/relation/DINO predictions.

        Token layout: [0,1]=current  [2,3]=future  [4]=goal  [5]=relation
        """
        current_tokens = z_student[:, 0:2, :]
        future_tokens  = z_student[:, 2:4, :]
        goal_tokens    = z_student[:, 4:5, :]
        relation_tokens = z_student[:, 5:6, :]

        current_logits = self.mask_decoder(current_tokens)
        future_logits  = self.mask_decoder(future_tokens)
        goal_logits    = self.mask_decoder(goal_tokens)

        # Relation head uses single token → squeeze
        rel_logits = self.relation_head(relation_tokens)

        pred_future_dino = self.dino_future_head(future_tokens)

        return SpatialTransitionOutput(
            z_student=z_student,
            current_tokens=current_tokens,
            future_tokens=future_tokens,
            goal_tokens=goal_tokens,
            relation_tokens=relation_tokens,
            current_mask_logits=current_logits,
            future_mask_logits=future_logits,
            goal_mask_logits=goal_logits,
            relation_logits=rel_logits,
            pred_future_dino=pred_future_dino,
        )

    def forward(self, vlm_hidden: torch.Tensor) -> SpatialTransitionOutput:
        """Full forward: encode + decode."""
        z = self.encode_student(vlm_hidden)
        return self.decode_student(z)


def build_spatial_backbone(
    vlm_dim: int = 2560,
    hidden_dim: int = 512,
    num_slots: int = 6,
    gamma: float = 1.5,
    **kwargs,
) -> SpatialTransitionBackbone:
    """Factory: build the formal SpatialTransitionBackbone."""
    return SpatialTransitionBackbone(
        vlm_dim=vlm_dim,
        hidden_dim=hidden_dim,
        num_slots=num_slots,
        gamma=gamma,
        **kwargs,
    )
