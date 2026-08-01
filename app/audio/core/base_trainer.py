"""Base trainer interface specification.

Provides the BaseAudioTrainer abstract class that model training engines must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class BaseAudioTrainer(ABC):
    """Abstract base class for audio model training engines."""

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Execute complete training loop across configured epochs.

        Returns:
            Dict[str, Any]: History metrics dictionary over all training epochs.
        """
        pass

    @abstractmethod
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute single training epoch iteration.

        Args:
            epoch (int): Current epoch number.

        Returns:
            Dict[str, float]: Metric results for the trained epoch (loss, accuracy).
        """
        pass

    @abstractmethod
    def validate(self) -> Dict[str, float]:
        """Execute validation pass over validation dataset.

        Returns:
            Dict[str, float]: Metric results over the validation dataset split.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, epoch: int, path: Optional[Path | str] = None) -> Path:
        """Persist trainer and model state checkpoint to disk.

        Args:
            epoch (int): Current epoch index.
            path (Optional[Path | str]): Target checkpoint file destination.

        Returns:
            Path: Written checkpoint file destination path.
        """
        pass
