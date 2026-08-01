"""Dataset configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from app.video.exceptions.video_exceptions import ConfigurationError


@dataclass
class DatasetConfig:
    """Configuration settings for video datasets."""

    dataset_name: str = "faceforensics_pp"
    data_dir: str = "datasets/faceforensics"
    split: str = "train"
    sequence_length: int = 16
    frame_stride: int = 1
    target_resolution: Tuple[int, int] = (224, 224)
    sampling_strategy: str = "uniform"
    metadata_path: Optional[str] = None
    max_samples: Optional[int] = None
    crop_faces: bool = True
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate dataset configuration."""
        if not self.dataset_name.strip():
            raise ConfigurationError("dataset_name cannot be empty")
        if self.sequence_length <= 0:
            raise ConfigurationError(
                f"sequence_length must be positive, got {self.sequence_length}"
            )
