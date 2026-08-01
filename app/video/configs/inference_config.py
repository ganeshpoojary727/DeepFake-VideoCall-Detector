"""Video inference configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from app.video.exceptions.video_exceptions import ConfigurationError


@dataclass
class VideoInferenceConfig:
    """Configuration settings for video inference pipelines."""

    target_fps: float = 30.0
    sequence_length: int = 16
    frame_stride: int = 1
    target_resolution: Tuple[int, int] = (224, 224)
    batch_size: int = 4
    confidence_threshold: float = 0.5
    device: str = "cuda"
    crop_faces: bool = True
    face_margin: float = 0.2
    normalize: bool = True
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate inference configuration parameters."""
        if self.sequence_length <= 0:
            raise ConfigurationError(
                f"sequence_length must be positive, got {self.sequence_length}"
            )
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ConfigurationError(
                f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}"
            )
