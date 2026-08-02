"""Video AI subsystem augmentations module exports."""

from app.video.augmentation.base_augmentation import BaseVideoAugmentation
from app.video.augmentation.spatial_aug import HorizontalFlip, RandomCrop, RandomRotation, Rotation
from app.video.augmentation.color_adjust import Brightness, Contrast, ColorJitter
from app.video.augmentation.blur_noise import Blur, GaussianBlur, GaussianNoise, Noise
from app.video.augmentation.compression import JPEG, JPEGCompression, RandomCompression
from app.video.augmentation.temporal_aug import TemporalDrop, FrameDropout, TemporalJitter, FrameSkip
from app.video.augmentation.augmentation_pipeline import VideoAugmentationPipeline

__all__ = [
    "BaseVideoAugmentation",
    "HorizontalFlip",
    "RandomCrop",
    "RandomRotation",
    "Rotation",
    "Brightness",
    "Contrast",
    "ColorJitter",
    "Blur",
    "GaussianBlur",
    "GaussianNoise",
    "Noise",
    "JPEG",
    "JPEGCompression",
    "RandomCompression",
    "TemporalDrop",
    "FrameDropout",
    "TemporalJitter",
    "FrameSkip",
    "VideoAugmentationPipeline",
]
