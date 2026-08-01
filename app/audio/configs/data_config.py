"""Audio data and dataset discovery configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from app.audio.constants.audio_constants import (
    AUDIO_CACHE_DIR,
    AUDIO_DATASETS_DIR,
    DATASET_ASVSPOOF_2019_LA,
    DATASET_ASVSPOOF_2021_LA,
    DATASET_ASVSPOOF_2021_DF,
)


@dataclass
class AudioDataConfig:
    """Configuration settings for production audio datasets and index building."""

    data_dir: Path = field(default_factory=lambda: Path(AUDIO_DATASETS_DIR))
    cache_dir: Path = field(default_factory=lambda: Path(AUDIO_CACHE_DIR))
    datasets: List[str] = field(
        default_factory=lambda: [
            DATASET_ASVSPOOF_2019_LA,
            DATASET_ASVSPOOF_2021_LA,
            DATASET_ASVSPOOF_2021_DF,
        ]
    )
    splits: List[str] = field(default_factory=lambda: ["train", "dev", "eval"])
    batch_size: int = 32
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        """Resolve directory paths."""
        self.data_dir = Path(self.data_dir)
        self.cache_dir = Path(self.cache_dir)
        self.validate()

    def validate(self) -> None:
        """Validate data config fields."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")
