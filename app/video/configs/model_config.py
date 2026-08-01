"""Video model architecture configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from app.video.exceptions.video_exceptions import ConfigurationError


@dataclass
class ModelConfig:
    """Configuration settings for video deepfake detector models."""

    model_name: str = "video_detector"
    backbone_name: str = "efficientnet_b4"
    attention_name: Optional[str] = "temporal_conv"
    classifier_name: str = "linear"
    num_classes: int = 2
    in_channels: int = 3
    feature_dim: int = 1792
    sequence_length: int = 16
    pretrained: bool = True
    dropout: float = 0.2
    input_resolution: Tuple[int, int] = (380, 380)
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate model configuration parameter integrity."""
        if self.num_classes <= 0:
            raise ConfigurationError(
                f"num_classes must be positive, got {self.num_classes}"
            )
        if self.in_channels <= 0:
            raise ConfigurationError(
                f"in_channels must be positive, got {self.in_channels}"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ConfigurationError(f"dropout must be in [0, 1), got {self.dropout}")
