"""Checkpoint manager for saving and pruning top-k and multi-criteria video model checkpoints."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from app.video.utils.checkpoint_utils import save_checkpoint

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages directory checkpoints for latest, best_loss, best_accuracy, and best_f1 checkpoints."""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = "checkpoints/video",
        max_to_keep: int = 3,
        save_top_k: int = 3,
        monitor: str = "val_loss",
        mode: str = "min",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_to_keep = save_top_k or max_to_keep
        self.save_top_k = self.max_to_keep
        self.monitor = monitor
        self.mode = mode.lower().strip()
        self._history: List[Tuple[float, str]] = []
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[Any] = None,
        epoch: int = 0,
        loss: float = 0.0,
        metrics: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Any] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """Save a checkpoint file to disk."""
        fname = filename or f"checkpoint_epoch_{epoch:03d}.pt"
        filepath = self.checkpoint_dir / fname

        save_checkpoint(
            filepath=str(filepath),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics or {"loss": loss},
        )
        logger.info(f"Saved checkpoint: {filepath}")
        return filepath

    def save(
        self,
        model: nn.Module,
        epoch: int,
        metrics: Dict[str, float],
        optimizer: Optional[Any] = None,
    ) -> Optional[str]:
        """Save model checkpoint with top-k pruning."""
        score = metrics.get(self.monitor, 0.0)
        filename = f"model_epoch_{epoch:03d}_{self.monitor}_{score:.4f}.pt"
        filepath = str(self.checkpoint_dir / filename)

        save_checkpoint(
            filepath=filepath,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
        )

        self._history.append((score, filepath))
        reverse = self.mode == "max"
        self._history.sort(key=lambda x: x[0], reverse=reverse)

        if len(self._history) > self.save_top_k:
            pruned_score, pruned_path = self._history.pop()
            if os.path.exists(pruned_path) and pruned_path != filepath:
                try:
                    os.remove(pruned_path)
                except OSError:
                    pass

        return filepath
