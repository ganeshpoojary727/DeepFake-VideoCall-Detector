"""Dataset configuration dataclass module for video AI subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.video.constants.video_constants import (
    DATASET_FACEFORENSICS,
    DEFAULT_FRAME_RATE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_SEQUENCE_LENGTH,
    VIDEO_CACHE_DIR,
    VIDEO_DATASETS_DIR,
)
from app.video.exceptions.video_exceptions import ConfigurationError


@dataclass
class DatasetConfig:
    """Configuration settings for video dataset discovery, metadata, and loading."""

    dataset_name: str = "faceforensics_pp"
    data_dir: Path = field(default_factory=lambda: Path(VIDEO_DATASETS_DIR))
    cache_dir: Path = field(default_factory=lambda: Path(VIDEO_CACHE_DIR))
    splits: List[str] = field(default_factory=lambda: ["train", "val", "test"])
    split: str = "train"
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH
    frame_count: int = DEFAULT_SEQUENCE_LENGTH
    fps: float = DEFAULT_FRAME_RATE
    frame_stride: int = 1
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE
    target_resolution: Tuple[int, int] = DEFAULT_IMAGE_SIZE
    batch_size: int = 8
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True
    sampling_strategy: str = "uniform"
    metadata_path: Optional[str] = None
    max_samples: Optional[int] = None
    crop_faces: bool = True
    use_cache: bool = True
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Resolve paths."""
        self.data_dir = Path(self.data_dir)
        self.cache_dir = Path(self.cache_dir)

    def validate(self) -> None:
        """Validate dataset configuration fields."""
        if not self.dataset_name or not str(self.dataset_name).strip():
            raise ConfigurationError("dataset_name cannot be empty")
        if self.sequence_length <= 0:
            raise ConfigurationError(
                f"sequence_length must be positive, got {self.sequence_length}"
            )
        if self.batch_size <= 0:
            raise ConfigurationError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_workers < 0:
            raise ConfigurationError(f"num_workers must be non-negative, got {self.num_workers}")


# Config dataclass alias
VideoDataConfig = DatasetConfig
