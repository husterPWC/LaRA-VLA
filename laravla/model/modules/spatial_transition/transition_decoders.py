"""
Transition decoders (bottleneck): predict mask & relation from transition tokens.
==================================================================================
Input:  transition_tokens [B, Kt, transition_dim]
Output: future_mask_logits [B, 1, 56, 56]
        goal_mask_logits   [B, 1, 56, 56]
        relation_logits    [B, num_classes]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskDecoder(nn.Module):
    """
    Decode transition tokens into mask logits.
    Uses mean pooling over tokens → token-count agnostic (works with 1, 2, or 6 tokens).
    """

    def __init__(
        self,
        transition_dim: int = 512,
        num_transition_tokens: int = 6,   # kept for API compat, unused
        output_res: int = 56,
    ):
        super().__init__()
        self.output_res = output_res
        self.transition_dim = transition_dim
        # Mean-pool over tokens first, then decode from fixed dim
        self.pool = nn.Sequential(
            nn.Linear(transition_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),  # 8
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  nn.BatchNorm2d(32), nn.ReLU(inplace=True),  # 16
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  nn.BatchNorm2d(16), nn.ReLU(inplace=True),  # 32
            nn.ConvTranspose2d(16, 1, 4, 2, 1),                                            # 64→crop
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, transition_tokens):
        """
        Args:
            transition_tokens: [B, T, D] where T can be any number of tokens

        Returns:
            mask_logits: [B, 1, output_res, output_res]
        """
        B = transition_tokens.shape[0]
        # Mean pool over tokens → [B, D]
        x = transition_tokens.mean(dim=1)
        feat = self.pool(x).view(B, 128, 4, 4)
        out = self.upsample(feat)
        if out.shape[-1] != self.output_res:
            diff = out.shape[-1] - self.output_res
            out = out[..., diff//2:diff//2+self.output_res, diff//2:diff//2+self.output_res]
        return out


class RelationHead(nn.Module):
    """Classify spatial relation from transition tokens. Mean-pool over tokens."""

    def __init__(
        self,
        transition_dim: int = 512,
        num_transition_tokens: int = 6,   # kept for API compat, unused
        num_classes: int = 6,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(transition_dim, 128),  # mean-pool over tokens first
            nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, transition_tokens):
        """[B, T, D] → [B, num_classes]"""
        return self.classifier(transition_tokens.mean(dim=1))


class TransitionToActionProjector(nn.Module):
    """Project transition tokens (mean-pooled) back to VLM hidden dim for action head."""

    def __init__(
        self,
        transition_dim: int = 512,
        num_transition_tokens: int = 6,   # kept for API compat, unused
        vlm_dim: int = 2560,
    ):
        super().__init__()
        self.project = nn.Linear(transition_dim, vlm_dim)  # mean-pool first

    def forward(self, transition_tokens):
        """[B, Kt, Dt] → mean pool → [B, Dvlm]"""
        return self.project(transition_tokens.mean(dim=1))
