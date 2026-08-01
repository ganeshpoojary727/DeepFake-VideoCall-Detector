"""Video data augmentation configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from app.video.exceptions.video_exceptions import ConfigurationError


@dataclass
class AugmentationConfig:
    """Configuration settings for video frame sequence augmentations."""

    enable_augmentation: bool = True
    jpeg_quality_range: Tuple[int, int] = (50, 95)
    blur_kernel_size: int = 5
    noise_std: float = 0.05
    brightness_factor: float = 0.2
    contrast_factor: float = 0.2
    color_jitter_params: Tuple[float, float, float] = (0.2, 0.2, 0.2)
    temporal_drop_prob: float = 0.1
    crop_size: Tuple[int, int] = (224, 224)
    horizontal_flip_prob: float = 0.5
    frame_skip_rate: int = 2
    gaussian_noise_prob: float = 0.3
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
