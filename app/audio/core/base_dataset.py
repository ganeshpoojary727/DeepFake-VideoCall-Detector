"""Base dataset interface specification for ASVspoof and AASIST audio models.

Provides the BaseAudioDataset abstract class that all audio PyTorch datasets implement,
enforcing standard audio tensor formats, metadata introspection, and label mappings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from app.audio.constants.audio_constants import (
    DEFAULT_NUM_SAMPLES,
    DEFAULT_SAMPLE_RATE,
    LABEL_BONAFIDE,
    LABEL_SPOOF,
)


class BaseAudioDataset(ABC, Dataset):
    """Abstract base class for audio PyTorch datasets.

    Standardizes dataset properties for raw waveform (e.g. 64,600 samples @ 16kHz for AASIST)
    and spectrogram / feature representations.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        target_samples: int = DEFAULT_NUM_SAMPLES,
        return_raw: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.target_samples = target_samples
        self.return_raw = return_raw
        self.samples: List[Tuple[str, int]] = []
        self.metadata: List[Dict[str, Any]] = []

    @abstractmethod
    def __len__(self) -> int:
        """Get total number of audio samples.

        Returns:
            int: Number of audio samples in the dataset split.
        """
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Fetch audio feature/waveform tensor and ground truth label at index.

        Args:
            index (int): Sample index.

        Returns:
            Tuple[torch.Tensor, int]: Feature tensor and class label (0=bonafide, 1=spoof).
        """
        pass

    def get_label_distribution(self) -> Dict[int, int]:
        """Calculate sample counts per label class in the dataset split.

        Returns:
            Dict[int, int]: Dictionary mapping class labels (0, 1) to sample counts.
        """
        counts = {LABEL_BONAFIDE: 0, LABEL_SPOOF: 0}
        for _, label in self.samples:
            if label in counts:
                counts[label] += 1
            else:
                counts[label] = counts.get(label, 0) + 1
        return counts

    def get_sample_info(self, index: int) -> Dict[str, Any]:
        """Retrieve detailed metadata for sample at given index.

        Args:
            index (int): Sample index.

        Returns:
            Dict[str, Any]: Metadata dictionary.
        """
        if index < 0 or index >= len(self.samples):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.samples)}")

        if self.metadata and index < len(self.metadata):
            return dict(self.metadata[index])

        file_name, label = self.samples[index]
        return {
            "file_name": file_name,
            "label": label,
            "label_str": "bonafide" if label == LABEL_BONAFIDE else "spoof",
        }
