"""
Spatial Transition Module — Mask-Supervised Latent Spatial Transition Reasoning.
================================================================================
Formal unified architecture. P1 and P2 share a single SpatialTransitionBackbone.
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
    TransitionToActionProjector, GatedTransitionActionAdapter,
    ProprioEncoder, DINOSpatialProjector
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
from laravla.model.modules.spatial_transition.posterior_encoder import (
    PosteriorTransitionEncoder
)
# ── Formal unified architecture ────────────────────────────
from laravla.model.modules.spatial_transition.spatial_backbone import (
    SpatialTransitionBackbone, SpatialTransitionOutput, build_spatial_backbone
)
from laravla.model.modules.spatial_transition.dino_metric import (
    dino_future_cosine
)
