"""Audio processing and feature pipeline configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from app.audio.constants.audio_constants import DEFAULT_NUM_SAMPLES, DEFAULT_SAMPLE_RATE


@dataclass
class AudioPipelineConfig:
    """Configuration parameters for audio preprocessing and augmentation pipeline."""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    target_samples: int = DEFAULT_NUM_SAMPLES
    target_duration_seconds: float = 4.0
    normalize: bool = True
    trim_silence: bool = False
    enable_augmentation: bool = True
    device: str = "cpu"

    def validate(self) -> None:
        """Validate pipeline parameters."""
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.target_samples <= 0:
            raise ValueError(f"target_samples must be positive, got {self.target_samples}")
