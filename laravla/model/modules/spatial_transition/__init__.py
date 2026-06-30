"""
Spatial Transition Module — Mask-Conditioned Latent Transition Reasoning (bottleneck).
======================================================================================
Lightweight bottleneck architecture (default 512-dim):
  VLM hidden (2560) → VLMProjector → (512)
  Mask → MaskTokenEncoder → [B,K,512]
  [VLM_proj + mask_tokens] → TransitionModule → [B,Kt,512]
  transition_tokens → MaskDecoders, RelationHead
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
from laravla.model.modules.spatial_transition.transition_losses import (
    mask_loss, relation_loss, transition_total_loss
)
