"""Base trainer interface specification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class BaseTrainer(ABC):
    """Abstract base class for video model training engines."""

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Execute complete model training workflow across configured epochs.

        Returns:
            Dict[str, Any]: Training history metrics dictionary across all epochs.
        """
        pass

    @abstractmethod
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute single epoch training iteration.

        Args:
            epoch: Current epoch index.

        Returns:
            Dict[str, float]: Metrics evaluated over epoch (loss, accuracy).
        """
        pass

    @abstractmethod
    def validate(self) -> Dict[str, float]:
        """Execute evaluation pass over validation dataset split.

        Returns:
            Dict[str, float]: Validation metric evaluation results.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, epoch: int, path: Optional[Path | str] = None) -> Path:
        """Persist model and trainer state checkpoint to disk.

        Args:
            epoch: Current epoch index.
            path: Target file path.

        Returns:
            Path: Written checkpoint file path.
        """
        pass


# Base class alias
BaseVideoTrainer = BaseTrainer
