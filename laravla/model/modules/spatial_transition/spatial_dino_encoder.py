"""
Spatial DINO Encoder — frozen DINOv2 for dense spatial features.
===================================================================
Extracts patch-level features from RGB images using a frozen DINOv2
backbone. These features capture dense spatial structure (object
boundaries, part layout, semantic correspondence) that complements
VLM semantic features.

Default: DINOv2 ViT-B/14 → 768-dim patch tokens.
ViT-S/14 (384-dim) kept as fast debug option.

Used by:
  - P1: DINO future feature prediction loss (auxiliary supervision)
  - P2: spatial-transition stream in action expert
  - Teacher posterior: current + future DINO features → transition target
  - Visualization: DINO similarity heatmaps

Reference: LaWAM uses frozen DINOv3 ViT-B/16 for the same purpose.

Design:
  - DINO encoder: fully frozen (no_grad always), no trainable parameters
  - DINO projector: trainable, projects DINO dim → transition dim
  - Strips CLS token, returns patch tokens [B, K, dino_dim]
  - Handles ImageNet normalization internally
  - All downstream code reads dino_dim from encoder.embed_dim (no hardcoding)

Usage:
    dino = build_dino_encoder("dinov2_vitb14", pretrained=True)
    dino_proj = DINOProjector(dino_dim=768, out_dim=512)
    tokens = dino(images)           # [B, 3, 224, 224] → [B, 256, 768]
    proj   = dino_proj(tokens)      # [B, 256, 768] → [B, 256, 512]
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import warnings

# ImageNet stats for DINOv2 input normalization
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Available DINOv2 models and their specs
_DINOV2_SPECS = {
    "dinov2_vits14":  {"embed_dim": 384,  "patch_size": 14},
    "dinov2_vitb14":  {"embed_dim": 768,  "patch_size": 14},
    "dinov2_vitl14":  {"embed_dim": 1024, "patch_size": 14},
    "dinov2_vitg14":  {"embed_dim": 1536, "patch_size": 14},
}

# Default model for formal experiments
_DEFAULT_DINO_MODEL = "dinov2_vitb14"


class SpatialDINOEncoder(nn.Module):
    """
    Frozen DINOv2 encoder for dense spatial feature extraction.

    Default: DINOv2 ViT-B/14 (768-dim). Use ViT-S/14 (384-dim) for fast debug.

    Args:
        model_name: DINOv2 variant ('dinov2_vitb14', 'dinov2_vits14', 'dinov2_vitl14')
        freeze: Always True — this encoder is never trained
        normalize: Apply ImageNet normalization internally (default True)
        cache_dir: Optional torch.hub cache directory
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_DINO_MODEL,
        freeze: bool = True,
        normalize: bool = True,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()

        if model_name not in _DINOV2_SPECS:
            raise ValueError(
                f"Unknown DINOv2 model: {model_name}. "
                f"Choose from: {list(_DINOV2_SPECS.keys())}"
            )

        spec = _DINOV2_SPECS[model_name]
        self.embed_dim = spec["embed_dim"]    # dino_dim: 768 for ViT-B, 384 for ViT-S
        self.patch_size = spec["patch_size"]   # always 14
        self.model_name = model_name
        self._do_normalize = normalize

        # Load from torch.hub (facebookresearch/dinov2)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*xFormers.*")
            if cache_dir is not None:
                torch.hub.set_dir(cache_dir)
            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                model_name,
                pretrained=False,
                skip_validation=True,
            )

        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()

        # Register normalization buffer
        self.register_buffer(
            "imagenet_mean",
            torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    # ── Public API ────────────────────────────────────────────

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract patch tokens from RGB images.

        Args:
            images: [B, 3, H, W] uint8 or float32 RGB images

        Returns:
            features: [B, K, dino_dim] normalized patch tokens (CLS removed)
              - K = (H/patch_size) * (W/patch_size)
              - dino_dim = self.embed_dim (768 for ViT-B/14, 384 for ViT-S/14)

        Example:
            >>> enc = SpatialDINOEncoder("dinov2_vitb14")
            >>> img = torch.rand(2, 3, 224, 224)
            >>> feat = enc(img)
            >>> feat.shape  # torch.Size([2, 256, 768])
        """
        if self._do_normalize:
            images = self._normalize(images)

        with torch.no_grad():
            outputs = self.model.forward_features(images)

        # Extract patch tokens only (no CLS)
        # DINOv2 returns 'x_norm_patchtokens' with shape [B, K, dino_dim]
        return outputs["x_norm_patchtokens"]

    def get_feature_map(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features as spatial feature map.

        Args:
            images: [B, 3, H, W]

        Returns:
            feature_map: [B, dino_dim, H_grid, W_grid]
        """
        tokens = self.forward(images)  # [B, K, dino_dim]
        B, K, D = tokens.shape
        grid = int(K ** 0.5)
        return tokens.transpose(1, 2).reshape(B, D, grid, grid)

    def get_patch_grid(self, image_size: int = 224) -> Tuple[int, int]:
        """Return (H_grid, W_grid) for a given image size."""
        g = image_size // self.patch_size
        return (g, g)

    @property
    def num_patches(self) -> int:
        """Number of patches for 224×224 input."""
        return (224 // self.patch_size) ** 2

    def num_patches_for_size(self, image_size: int) -> int:
        """Number of patches for a given square image size."""
        return (image_size // self.patch_size) ** 2

    def load_pretrained(self, progress: bool = True):
        """Download and load pretrained DINOv2 weights."""
        state_dict = torch.hub.load_state_dict_from_url(
            f"https://dl.fbaipublicfiles.com/dinov2/{self.model_name}/{self.model_name}_pretrain.pth",
            progress=progress,
        )
        self.model.load_state_dict(state_dict, strict=True)
        return self

    # ── Internal ──────────────────────────────────────────────

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        """Convert uint8 [0,255] or float [0,1] → ImageNet-normalized float."""
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        elif images.max() > 1.5:
            images = images / 255.0

        if images.dim() == 3:
            images = images.unsqueeze(0)

        mean = self.imagenet_mean.to(images.device)
        std = self.imagenet_std.to(images.device)
        return (images - mean) / std


class DINOProjector(nn.Module):
    """
    Trainable projector: DINO feature dim → transition dim.

    DINO encoder is frozen; this projector is trained to map dense spatial
    features into the transition bottleneck space.

    Architecture: LayerNorm → Linear → GELU → Linear
    """

    def __init__(self, dino_dim: int, out_dim: int = 512):
        super().__init__()
        self.dino_dim = dino_dim
        self.out_dim = out_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(dino_dim),
            nn.Linear(dino_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, K, dino_dim] DINO patch tokens

        Returns:
            [B, K, out_dim] projected features
        """
        return self.proj(x)


# ── Factory ───────────────────────────────────────────────────

def build_dino_encoder(
    model_name: str = _DEFAULT_DINO_MODEL,
    pretrained: bool = False,
    image_size: int = 224,
    **kwargs,
) -> SpatialDINOEncoder:
    """
    Factory: build a frozen DINOv2 encoder.

    Args:
        model_name: 'dinov2_vitb14' (default), 'dinov2_vits14' (debug), 'dinov2_vitl14'
        pretrained: If True, download pretrained weights (requires internet)
        image_size: Expected input size (for metadata only)
        **kwargs: Passed to SpatialDINOEncoder

    Returns:
        SpatialDINOEncoder (frozen, eval mode)
    """
    encoder = SpatialDINOEncoder(model_name=model_name, freeze=True, **kwargs)
    if pretrained:
        encoder.load_pretrained()
    return encoder
