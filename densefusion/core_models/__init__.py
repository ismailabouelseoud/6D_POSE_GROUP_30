"""Core model implementations (custom transformer + encoders).

These are lightweight, self-contained modules that demonstrate a
from-scratch Multi-Head Self-Attention (MHSA) and a small transformer
encoder used for RGB-depth fusion. They are intentionally compact so
they can be reviewed quickly by hiring managers.
"""

from .transformer_fusion import MultiHeadSelfAttention, TransformerEncoderBlock, TransformerFuserCustom
from .rgb_encoder import RGBEncoder
from .pointnet import PointNetEncoder
from .pose_estimator import PoseEstimator

__all__ = [
    'MultiHeadSelfAttention', 'TransformerEncoderBlock', 'TransformerFuserCustom',
    'RGBEncoder', 'PointNetEncoder', 'PoseEstimator'
]
