"""Base dataset interface specification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    """Abstract base class for video PyTorch datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """Get total sample count in dataset split.

        Returns:
            int: Number of video samples.
        """
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Fetch video sample at specified dataset index.

        Args:
            index: Dataset index.

        Returns:
            Any: Sample object or tuple of (tensor, label).
        """
        pass

    @abstractmethod
    def get_label_distribution(self) -> Dict[int, int]:
        """Calculate sample distribution count per target class label.

        Returns:
            Dict[int, int]: Mapping of class label to sample count.
        """
        pass


# Base class alias
BaseVideoDataset = BaseDataset
