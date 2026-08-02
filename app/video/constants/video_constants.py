"""Video AI subsystem constant definitions.

Provides centralized constants for spatial resolutions, sequence lengths,
image normalization defaults, production video dataset identifiers, supported video/image
file formats, and directory paths.
"""

from __future__ import annotations

from typing import Tuple

# Spatial Dimensions & Resolution
DEFAULT_FRAME_HEIGHT: int = 224
DEFAULT_FRAME_WIDTH: int = 224
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH)
EFFICIENTNET_B4_INPUT_SIZE: int = 380

# Temporal Sequence Parameters
DEFAULT_SEQUENCE_LENGTH: int = 16
DEFAULT_FRAME_RATE: float = 30.0
DEFAULT_FPS: float = DEFAULT_FRAME_RATE
DEFAULT_FRAME_STRIDE: int = 1

# Image Normalization Defaults (ImageNet Standards)
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Supported Video & Image File Extensions
SUPPORTED_IMAGE_FORMATS: Tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)

SUPPORTED_VIDEO_FORMATS: Tuple[str, ...] = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
)

# Production Paths
VIDEO_DATASETS_DIR: str = "datasets/video"
VIDEO_CACHE_DIR: str = "datasets/video/cache"
VIDEO_LOGS_DIR: str = "logs/video"
VIDEO_MODELS_DIR: str = "trained_models/video"

# Production Dataset Identifiers
DATASET_FACEFORENSICS: str = "faceforensics_pp"
DATASET_CELEBDFV2: str = "celeb_df_v2"
DATASET_FFPP: str = "faceforensics_pp"
DATASET_CELEB_DF: str = "celeb_df_v2"
DATASET_DEEPFORENSICS: str = "deepforensics"

SUPPORTED_VIDEO_DATASETS: Tuple[str, ...] = (
    DATASET_FACEFORENSICS,
    DATASET_CELEBDFV2,
)

SUPPORTED_DATASETS: Tuple[str, ...] = (
    DATASET_FFPP,
    DATASET_CELEB_DF,
    DATASET_DEEPFORENSICS,
)

# FaceForensics++ Manipulation Categories
FFPP_ORIGINAL: str = "Original"
FFPP_FACE2FACE: str = "Face2Face"
FFPP_FACESWAP: str = "FaceSwap"
FFPP_FACESHIFTER: str = "FaceShifter"
FFPP_DEEPFAKES: str = "Deepfakes"
FFPP_NEURALTEXTURES: str = "NeuralTextures"
FFPP_DEEPFAKEDETECTION: str = "DeepFakeDetection"

FFPP_CATEGORIES: Tuple[str, ...] = (
    FFPP_ORIGINAL,
    FFPP_FACE2FACE,
    FFPP_FACESWAP,
    FFPP_FACESHIFTER,
    FFPP_DEEPFAKES,
    FFPP_NEURALTEXTURES,
    FFPP_DEEPFAKEDETECTION,
)

# CelebDF-v2 Categories
CELEBDF_REAL: str = "Celeb-real"
CELEBDF_SYNTHESIS: str = "Celeb-synthesis"
CELEBDF_YOUTUBE_REAL: str = "YouTube-real"

CELEBDF_CATEGORIES: Tuple[str, ...] = (
    CELEBDF_REAL,
    CELEBDF_SYNTHESIS,
    CELEBDF_YOUTUBE_REAL,
)

# Model Backbones & Attention Keys
BACKBONE_EFFICIENTNET_B4: str = "efficientnet_b4"
BACKBONE_RESNET50: str = "resnet50"
SUPPORTED_BACKBONES: Tuple[str, ...] = (
    BACKBONE_EFFICIENTNET_B4,
    BACKBONE_RESNET50,
)

ATTENTION_TEMPORAL_CONV: str = "temporal_conv"
ATTENTION_TRANSFORMER: str = "temporal_transformer"
SUPPORTED_ATTENTIONS: Tuple[str, ...] = (
    ATTENTION_TEMPORAL_CONV,
    ATTENTION_TRANSFORMER,
)

# Default Device Specification
DEFAULT_DEVICE: str = "cuda"
FALLBACK_DEVICE: str = "cpu"
