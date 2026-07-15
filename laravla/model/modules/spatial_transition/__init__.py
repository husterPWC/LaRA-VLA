"""
Spatial Transition Module — Mask-Supervised Latent Spatial Transition Reasoning.
================================================================================
Bottleneck architecture (default 512-dim):
  RGB → VLM hidden (2560) → VLMProjector → (512)
  RGB → frozen DINO → dense spatial features (384/768)
  VLM_proj → TransitionModule → [B,Kt,512]
  transition_tokens → MaskDecoders (current/future/goal), RelationHead
  transition_tokens → DINO future predictor (auxiliary)
"""

from laravla.model.modules.spatial_transition.mask_token_encoder import (
    MaskTokenEncoder, VLMProjector
)
from laravla.model.modules.spatial_transition.transition_module import (
    MaskConditionedTransitionModule
)
from laravla.model.modules.spatial_transition.transition_decoders import (
    MaskDecoder, RelationHead, TransitionToActionProjector
)
from laravla.model.modules.spatial_transition.transition_action_adapter import (
    TransitionToActionProjector, GatedTransitionActionAdapter
)
from laravla.model.modules.spatial_transition.p1_wrapper import (
    P1TransitionWrapper, P1NoMaskWrapper
)
from laravla.model.modules.spatial_transition.transition_losses import (
    mask_loss, relation_loss, transition_total_loss, token_diversity_loss
)
from laravla.model.modules.spatial_transition.spatial_dino_encoder import (
    SpatialDINOEncoder, DINOProjector, build_dino_encoder
)
from laravla.model.modules.spatial_transition.dino_future_head import (
    DINOFutureHead, dino_cosine_loss, dino_cosine_similarity
)
