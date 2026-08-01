"""
Checkpoint saving, loading, top-k management, and integrity checking module.

Provides the CheckpointManager class for persisting and restoring model weights,
optimizer states, scheduler states, and metrics.
"""

from __future__ import annotations

import heapq
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer

from app.audio.utils.logger import AudioLogger

logger = AudioLogger.get("training.checkpoint")


class CheckpointManager:
    """
    Manager for model checkpoint serialization, deserialization, and top-k retention.

    Parameters
    ----------
    checkpoint_dir : Path | str
        Directory where checkpoint files are persisted.
    max_to_keep : int
        Maximum number of top-performing checkpoints to retain.
    """

    def __init__(self, checkpoint_dir: Path | str, max_to_keep: int = 3) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_to_keep = max_to_keep
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._top_checkpoints: List[Tuple[float, Path]] = []

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metric_value: Optional[float] = None,
        filename: str = "checkpoint.pth",
    ) -> Path:
        """
        Save model state, optimizer, scheduler, and metadata to disk.

        Parameters
        ----------
        model : nn.Module
            PyTorch neural network model.
        optimizer : Optional[Optimizer]
            PyTorch optimizer instance.
        scheduler : Optional[Any]
            Learning rate scheduler instance.
        epoch : int
            Current training epoch index.
        metric_value : Optional[float]
            Validation metric (e.g. loss or EER) for top-k tracking.
        filename : str
            Target filename for the checkpoint.

        Returns
        -------
        Path
            Path to the written checkpoint file.
        """
        save_path = self.checkpoint_dir / filename
        state_dict: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metric_value": metric_value,
        }
        if optimizer is not None:
            state_dict["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            state_dict["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(state_dict, save_path)
        logger.info("Saved checkpoint to %s (epoch=%d)", save_path, epoch)

        if metric_value is not None and self.max_to_keep > 0:
            self._manage_top_k(metric_value, save_path)

        return save_path

    def load(
        self,
        checkpoint_path: Path | str,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Load checkpoint weights and state dicts into model and optimizer.

        Parameters
        ----------
        checkpoint_path : Path | str
            Path to checkpoint file.
        model : nn.Module
            Model to receive weights.
        optimizer : Optional[Optimizer]
            Optimizer to receive state dict.
        scheduler : Optional[Any]
            Scheduler to receive state dict.
        device : Optional[torch.device]
            Target hardware compute device.

        Returns
        -------
        Dict[str, Any]
            Loaded checkpoint metadata dictionary.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded model weights from %s", path)

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint

    def _manage_top_k(self, metric_value: float, checkpoint_path: Path) -> None:
        """Track top-k checkpoints based on metric value (lower is better)."""
        heapq.heappush(self._top_checkpoints, (-metric_value, checkpoint_path))
        if len(self._top_checkpoints) > self.max_to_keep:
            _, oldest_path = heapq.heappop(self._top_checkpoints)
            if oldest_path.exists() and oldest_path != checkpoint_path:
                try:
                    oldest_path.unlink()
                    logger.info("Pruned old checkpoint: %s", oldest_path)
                except OSError as err:
                    logger.warning("Could not prune checkpoint %s: %s", oldest_path, err)
