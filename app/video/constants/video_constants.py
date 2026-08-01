"""Video AI subsystem constant definitions.

Centralized constants for spatial resolution, sequence lengths, image norm,
dataset identifiers, model keys, and default hyperparameter defaults.
"""

from __future__ import annotations

# Spatial Dimensions
DEFAULT_FRAME_HEIGHT: int = 224
DEFAULT_FRAME_WIDTH: int = 224
EFFICIENTNET_B4_INPUT_SIZE: int = 380

# Temporal Sequence Dimensions
DEFAULT_SEQUENCE_LENGTH: int = 16
DEFAULT_FRAME_RATE: float = 30.0
DEFAULT_FRAME_STRIDE: int = 1

# Image Normalization Defaults (ImageNet Standards)
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

# Supported Dataset Keys
DATASET_FFPP: str = "faceforensics_pp"
DATASET_CELEB_DF: str = "celeb_df_v2"
DATASET_DEEPFORENSICS: str = "deepforensics"
SUPPORTED_DATASETS: tuple[str, ...] = (
    DATASET_FFPP,
    DATASET_CELEB_DF,
    DATASET_DEEPFORENSICS,
)

# Supported Backbone Model Identifiers
BACKBONE_EFFICIENTNET_B4: str = "efficientnet_b4"
BACKBONE_RESNET50: str = "resnet50"
SUPPORTED_BACKBONES: tuple[str, ...] = (
    BACKBONE_EFFICIENTNET_B4,
    BACKBONE_RESNET50,
)

# Supported Temporal Attention Types
ATTENTION_TEMPORAL_CONV: str = "temporal_conv"
ATTENTION_TRANSFORMER: str = "temporal_transformer"
SUPPORTED_ATTENTIONS: tuple[str, ...] = (
    ATTENTION_TEMPORAL_CONV,
    ATTENTION_TRANSFORMER,
)

# Default Device Specification
DEFAULT_DEVICE: str = "cuda"
FALLBACK_DEVICE: str = "cpu"
