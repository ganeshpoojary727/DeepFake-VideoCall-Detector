"""Base dataset interface specification.

Provides the BaseAudioDataset abstract class that all audio PyTorch datasets
must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import Dataset


class BaseAudioDataset(ABC, Dataset):
    """Abstract base class for audio PyTorch datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """Get dataset size.

        Returns:
            int: Number of audio samples in the dataset split.
        """
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Fetch audio feature tensor and ground truth label at index.

        Args:
            index (int): Sample index.

        Returns:
            Tuple[torch.Tensor, int]: Feature tensor and class label tuple.
        """
        pass

    @abstractmethod
    def get_label_distribution(self) -> Dict[int, int]:
        """Calculate counts per label class in dataset split.

        Returns:
            Dict[int, int]: Dictionary mapping class labels to sample counts.
        """
        pass
