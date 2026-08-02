"""Video data augmentation configuration dataclass module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from app.video.exceptions.video_exceptions import ConfigurationError


@dataclass
class AugmentationConfig:
    """Configuration settings for independent spatial and temporal video data augmentations."""

    enable_augmentation: bool = True

    # 1. Random Crop
    enable_random_crop: bool = True
    crop_size: Tuple[int, int] = (224, 224)
    crop_prob: float = 0.5

    # 2. Horizontal Flip
    enable_horizontal_flip: bool = True
    horizontal_flip_prob: float = 0.5

    # 3. Color Jitter
    enable_color_jitter: bool = True
    brightness_factor: float = 0.2
    contrast_factor: float = 0.2
    color_jitter_prob: float = 0.5

    # 4. Blur
    enable_blur: bool = True
    blur_kernel_size: int = 5
    blur_prob: float = 0.3

    # 5. JPEG Compression
    enable_jpeg_compression: bool = True
    jpeg_quality_range: Tuple[int, int] = (50, 95)
    jpeg_prob: float = 0.4

    # 6. Noise
    enable_noise: bool = True
    noise_std: float = 0.05
    noise_prob: float = 0.3

    # 7. Rotation
    enable_rotation: bool = True
    max_rotation_degrees: float = 15.0
    rotation_prob: float = 0.3

    # 8. Frame Dropout
    enable_frame_dropout: bool = True
    frame_drop_prob: float = 0.1
    temporal_drop_prob: float = 0.3

    # 9. Temporal Jitter
    enable_temporal_jitter: bool = True
    temporal_jitter_max_shift: int = 2
    temporal_jitter_prob: float = 0.3

    extra_params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate augmentation configuration values."""
        min_q, max_q = self.jpeg_quality_range
        if min_q < 1 or max_q > 100 or min_q > max_q:
            raise ConfigurationError(
                f"Invalid JPEG quality range {self.jpeg_quality_range}"
            )
        if not (0.0 <= self.horizontal_flip_prob <= 1.0):
            raise ConfigurationError(
                f"horizontal_flip_prob must be in [0, 1], got {self.horizontal_flip_prob}"
            )


# Config dataclass alias
VideoAugmentationConfig = AugmentationConfig
