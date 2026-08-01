"""Video augmentation package."""

from app.video.augmentation.augmentation_pipeline import VideoAugmentationPipeline
from app.video.augmentation.base_augmentation import BaseVideoAugmentation
from app.video.augmentation.blur_noise import Blur, GaussianNoise, Noise
from app.video.augmentation.color_adjust import Brightness, ColorJitter, Contrast
from app.video.augmentation.compression import JPEG, RandomCompression
from app.video.augmentation.spatial_aug import HorizontalFlip, RandomCrop
from app.video.augmentation.temporal_aug import FrameSkip, TemporalDrop

__all__ = [
    "BaseVideoAugmentation",
    "JPEG",
    "RandomCompression",
    "Blur",
    "GaussianNoise",
    "Noise",
    "Brightness",
    "Contrast",
    "ColorJitter",
    "HorizontalFlip",
    "RandomCrop",
    "TemporalDrop",
    "FrameSkip",
    "VideoAugmentationPipeline",
]
