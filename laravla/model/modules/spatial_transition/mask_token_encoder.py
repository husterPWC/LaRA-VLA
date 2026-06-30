"""
MaskTokenEncoder + VLMProjector: lightweight mask-to-token encoding.
=====================================================================
VLMProjector:  Linear(2560 → transition_dim) — project frozen VLM hidden
MaskTokenEncoder: small CNN → bottleneck tokens [B, K, transition_dim]

Design: bottleneck at transition_dim (default 512), keeping adapter lightweight.
"""

import torch
import torch.nn as nn


class VLMProjector(nn.Module):
    """Project frozen VLM hidden states from 2560 to bottleneck transition_dim."""

    def __init__(self, vlm_dim: int = 2560, transition_dim: int = 512):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(vlm_dim, transition_dim),
            nn.LayerNorm(transition_dim),
        )

    def forward(self, vlm_hidden):
        """[B, L, 2560] → [B, L, transition_dim]"""
        return self.project(vlm_hidden)


class MaskTokenEncoder(nn.Module):
    """Encode binary mask into K bottleneck tokens."""

    def __init__(
        self,
        in_channels: int = 1,
        transition_dim: int = 512,
        num_tokens: int = 8,
        input_size: int = 224,
    ):
        super().__init__()
        self.transition_dim = transition_dim
        self.num_tokens = num_tokens

        # Small CNN: 224 → 112 → 56 → 28 → 14
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        # Project to num_tokens * transition_dim
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, transition_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(transition_dim * 2, transition_dim * num_tokens),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, agentview_mask, wrist_mask=None):
        """[B,1,224,224] → [B, K, transition_dim]"""
        B = agentview_mask.shape[0]
        feat = self.encoder(agentview_mask)
        feat = self.pool(feat)
        tokens = self.project(feat)
        return tokens.view(B, self.num_tokens, self.transition_dim)
