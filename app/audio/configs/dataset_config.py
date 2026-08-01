"""
Audio dataset and feature configuration definitions.

Provides dataclasses for configuring audio dataset paths, protocol parsing,
feature extraction parameters, and caching settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.config.settings import settings


@dataclass
class AudioFeatureConfig:
    """
    Configuration parameters for audio feature extraction.

    Parameters
    ----------
    sample_rate : int
        Target audio sampling rate in Hz.
    n_mels : int
        Number of Mel frequency bins.
    n_fft : int
        FFT window size.
    hop_length : int
        Hop length for STFT.
    target_length : int
        Target frame length after padding or cropping.
    """

    sample_rate: int = field(default_factory=lambda: settings.audio.sample_rate)
    n_mels: int = field(default_factory=lambda: settings.audio.n_mels)
    n_fft: int = field(default_factory=lambda: settings.audio.n_fft)
    hop_length: int = field(default_factory=lambda: settings.audio.hop_length)
    target_length: int = field(default_factory=lambda: settings.audio.target_length)

    def validate(self) -> None:
        """
        Validate feature extraction parameters.

        Raises
        ------
        ValueError
            If any configuration parameter is non-positive.
        """
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.n_mels <= 0:
            raise ValueError(f"n_mels must be positive, got {self.n_mels}")
        if self.n_fft <= 0:
            raise ValueError(f"n_fft must be positive, got {self.n_fft}")
        if self.hop_length <= 0:
            raise ValueError(f"hop_length must be positive, got {self.hop_length}")
        if self.target_length <= 0:
            raise ValueError(f"target_length must be positive, got {self.target_length}")


@dataclass
class DatasetConfig:
    """
    Configuration parameters for audio datasets.

    Parameters
    ----------
    dataset_dir : Path
        Root directory of the dataset.
    protocol_file : Optional[Path]
        Path to the protocol/label file (e.g., ASVspoof protocol).
    split : str
        Dataset split name ('train', 'dev', 'test').
    cache_in_memory : bool
        Whether to cache extracted features in memory.
    num_workers : int
        Number of worker subprocesses for data loading.
    feature_config : AudioFeatureConfig
        Sub-configuration for feature extraction.
    """

    dataset_dir: Path = field(default_factory=lambda: settings.DATASET_PATH)
    protocol_file: Optional[Path] = None
    split: str = "train"
    cache_in_memory: bool = False
    num_workers: int = field(default_factory=lambda: settings.training.num_workers)
    feature_config: AudioFeatureConfig = field(default_factory=AudioFeatureConfig)

    def __post_init__(self) -> None:
        """Ensure paths are Path instances and validate parameters."""
        self.dataset_dir = Path(self.dataset_dir)
        if self.protocol_file is not None:
            self.protocol_file = Path(self.protocol_file)
        self.feature_config.validate()

    def get_protocol_path(self) -> Path:
        """
        Resolve the protocol file path.

        Returns
        -------
        Path
            Explicit protocol file path or auto-resolved protocol path inside dataset_dir.
        """
        if self.protocol_file is not None:
            return self.protocol_file
        return self.dataset_dir / f"ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{self.split}.trn.txt"
