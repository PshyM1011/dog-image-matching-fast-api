"""Model definitions for dual-view dog image matching."""
from .dual_encoder import (
    DualViewEncoder,
    FrontalEncoder,
    LateralEncoder,
    DualViewFusionModel
)
from .loss import (
    TripletLoss,
    HardTripletLoss,
    ArcFaceLoss,
    CombinedLoss
)

__all__ = [
    'DualViewEncoder',
    'FrontalEncoder',
    'LateralEncoder',
    'DualViewFusionModel',
    'TripletLoss',
    'HardTripletLoss',
    'ArcFaceLoss',
    'CombinedLoss'
]

