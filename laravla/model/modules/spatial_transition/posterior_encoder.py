"""
PosteriorTransitionEncoder — Teacher branch for transition distillation.
==========================================================================
Training-only encoder that sees privileged information (DINO features,
masks, relation label) and produces z_teacher [B, 6, 512] with the same
typed token layout as the student.

The student (RGB + instruction → z_student) is trained to mimic the
teacher via MSE(LN(z_student), stopgrad(LN(z_teacher))).

Typed token layout (matching student):
  [0,1] = current  ← cur_mask + cur_dino
  [2,3] = future   ← fut_mask + fut_dino
  [4]   = goal     ← goal_mask
  [5]   = relation ← relation label
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PosteriorTransitionEncoder(nn.Module):
    """
    Encode privileged info → typed transition tokens.

    Inputs (training only):
        dino_cur:   [B, 256, 768]  current DINO patch features
        dino_fut:   [B, 256, 768]  tau future DINO patch features
        cur_mask:   [B, 1, H, W]   current affordance mask
        fut_mask:   [B, 1, H, W]   tau future affordance mask
        goal_mask:  [B, 1, H, W]   goal affordance mask
        rel_id:     [B] LongTensor relation label index

    Output:
        z_teacher:  [B, 6, transition_dim]
    """

    def __init__(self, dino_dim=768, transition_dim=512, mask_size=224,
                 num_relation_labels=6):
        super().__init__()
        self.transition_dim = transition_dim

        # Shared mask encoder: [1, H, W] → [transition_dim]
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),   # 112
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),   # 56
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),  # 28
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, transition_dim),
            nn.LayerNorm(transition_dim),
        )

        # DINO projector: mean-pool patches → [transition_dim]
        self.dino_projector = nn.Sequential(
            nn.Linear(dino_dim, transition_dim),
            nn.LayerNorm(transition_dim),
        )

        # Relation embedding
        self.relation_embed = nn.Embedding(num_relation_labels, transition_dim)

        # Type-specific projections: input_info → typed tokens
        # current: [cur_mask_feat + cur_dino_feat] → 2 tokens
        self.cur_proj = nn.Sequential(
            nn.Linear(transition_dim * 2, transition_dim * 2),
            nn.GELU(),
            nn.Linear(transition_dim * 2, transition_dim * 2),
        )
        # future: [fut_mask_feat + fut_dino_feat] → 2 tokens
        self.fut_proj = nn.Sequential(
            nn.Linear(transition_dim * 2, transition_dim * 2),
            nn.GELU(),
            nn.Linear(transition_dim * 2, transition_dim * 2),
        )
        # goal: [goal_mask_feat] → 1 token
        self.goal_proj = nn.Sequential(
            nn.Linear(transition_dim, transition_dim),
            nn.LayerNorm(transition_dim),
        )
        # relation: [rel_embed] → 1 token
        self.rel_proj = nn.Sequential(
            nn.Linear(transition_dim, transition_dim),
            nn.LayerNorm(transition_dim),
        )

        # Type embedding for output reinforcement
        self.type_embedding = nn.Parameter(
            torch.randn(1, 6, transition_dim) * 0.1
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.relation_embed.weight, std=0.02)

    def forward(self, dino_cur, dino_fut, cur_mask, fut_mask, goal_mask, rel_id):
        """
        Returns:
            z_teacher: [B, 6, transition_dim]
        """
        B = dino_cur.shape[0]

        # Encode masks → [B, transition_dim]
        cur_feat = self.mask_encoder(cur_mask)
        fut_feat = self.mask_encoder(fut_mask)
        goal_feat = self.mask_encoder(goal_mask)

        # Encode DINO (mean pool over patches) → [B, transition_dim]
        dino_cur_feat = self.dino_projector(dino_cur.mean(dim=1))
        dino_fut_feat = self.dino_projector(dino_fut.mean(dim=1))

        # Encode relation → [B, transition_dim]
        rel_feat = self.relation_embed(rel_id)

        # Type-specific projections → typed tokens
        cur_info = torch.cat([cur_feat, dino_cur_feat], dim=-1)   # [B, 2*D]
        fut_info = torch.cat([fut_feat, dino_fut_feat], dim=-1)   # [B, 2*D]

        cur_tokens = self.cur_proj(cur_info).view(B, 2, self.transition_dim)  # [B, 2, D]
        fut_tokens = self.fut_proj(fut_info).view(B, 2, self.transition_dim)  # [B, 2, D]
        goal_token = self.goal_proj(goal_feat).unsqueeze(1)                    # [B, 1, D]
        rel_token = self.rel_proj(rel_feat).unsqueeze(1)                       # [B, 1, D]

        z_teacher = torch.cat([cur_tokens, fut_tokens, goal_token, rel_token], dim=1)
        # [B, 6, D]

        # Add type embedding (same pattern as student's output)
        z_teacher = z_teacher + self.type_embedding
        return z_teacher
