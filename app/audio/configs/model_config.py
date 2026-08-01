"""Audio model architecture configuration definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from app.config.settings import settings


@dataclass
class AudioModelConfig:
    """Configuration parameters for audio deepfake classification models."""

    model_name: str = field(default_factory=lambda: settings.model.model_name)
    num_classes: int = field(default_factory=lambda: settings.model.num_classes)
    in_channels: int = 1
    feature_dim: int = 128
    dropout: float = 0.3
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    confidence_threshold: float = field(
        default_factory=lambda: settings.model.confidence_threshold
    )

    def __post_init__(self) -> None:
        """Validate architecture parameters."""
        self.validate()

    def validate(self) -> None:
        """Validate architecture configuration parameters."""
        if self.num_classes <= 1:
            raise ValueError(f"num_classes must be > 1, got {self.num_classes}")
        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {self.in_channels}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")
        if not (0.0 < self.confidence_threshold < 1.0):
            raise ValueError(
                f"confidence_threshold must be in (0.0, 1.0), got {self.confidence_threshold}"
            )
        if not self.conv_channels:
            raise ValueError("conv_channels cannot be empty")


# Alias for backward compatibility
ModelConfig = AudioModelConfig
