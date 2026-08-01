"""Checkpoint manager for saving and pruning top-k video model checkpoints."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
import torch.nn as nn

from app.video.utils.checkpoint_utils import save_checkpoint


class CheckpointManager:
    """Manages directory checkpoints keeping top-k best performing model weights."""

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints/video",
        save_top_k: int = 3,
        monitor: str = "val_loss",
        mode: str = "min",
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode.lower().strip()
        self._history: List[Tuple[float, str]] = []  # (score, path)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        epoch: int,
        metrics: Dict[str, float],
        optimizer: Optional[Any] = None,
    ) -> Optional[str]:
        """Save model checkpoint if it ranks within top-k saved checkpoints.

        Args:
            model: PyTorch model module.
            epoch: Epoch index.
            metrics: Metrics dict containing monitored key.
            optimizer: Optional optimizer instance.

        Returns:
            Optional[str]: Saved file path string if saved, else None.
        """
        score = metrics.get(self.monitor, 0.0)
        filename = f"model_epoch_{epoch:03d}_{self.monitor}_{score:.4f}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)

        save_checkpoint(
            filepath=filepath,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
        )

        self._history.append((score, filepath))

        # Sort history: ascending if min mode (lower is better), descending if max mode
        reverse = self.mode == "max"
        self._history.sort(key=lambda x: x[0], reverse=reverse)

        # Prune older checkpoints beyond top-k
        if len(self._history) > self.save_top_k:
            pruned_score, pruned_path = self._history.pop()
            if os.path.exists(pruned_path) and pruned_path != filepath:
                try:
                    os.remove(pruned_path)
                except OSError:
                    pass

        return filepath
