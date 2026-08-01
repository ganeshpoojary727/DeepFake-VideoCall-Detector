"""Checkpoint saving, loading, top-k management, and artifact production module."""

from __future__ import annotations

import heapq
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer

from app.audio.constants.audio_constants import AUDIO_MODELS_DIR


class CheckpointManager:
    """Manager for model checkpoint serialization, deserialization, and artifact output."""

    def __init__(self, checkpoint_dir: Path | str = AUDIO_MODELS_DIR, max_to_keep: int = 3) -> None:
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
        filename: str = "checkpoint.pt",
        history: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save model state, optimizer, scheduler, and metadata to disk."""
        save_path = self.checkpoint_dir / filename
        state_dict: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metric_value": metric_value,
        }
        if optimizer is not None:
            state_dict["optimizer_state_dict"] = optimizer.state_dict()
            torch.save(optimizer.state_dict(), self.checkpoint_dir / "optimizer.pt")
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            state_dict["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(scheduler.state_dict(), self.checkpoint_dir / "scheduler.pt")

        torch.save(state_dict, save_path)
        # Always save as last_checkpoint.pt
        torch.save(state_dict, self.checkpoint_dir / "last_checkpoint.pt")

        if history is not None:
            try:
                with open(self.checkpoint_dir / "training_history.json", "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)
            except Exception:
                pass

        if metric_value is not None and self.max_to_keep > 0:
            self._manage_top_k(metric_value, save_path)

        return save_path

    def save_best(self, model: nn.Module, epoch: int, metrics: Dict[str, float]) -> Path:
        """Save best model state as best_model.pt."""
        best_path = self.checkpoint_dir / "best_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
            },
            best_path,
        )
        return best_path

    def load(
        self,
        checkpoint_path: Path | str,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Load checkpoint weights and state dicts into model and optimizer."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint

    def _manage_top_k(self, metric_value: float, checkpoint_path: Path) -> None:
        """Track top-k checkpoints based on metric value."""
        heapq.heappush(self._top_checkpoints, (-metric_value, checkpoint_path))
        if len(self._top_checkpoints) > self.max_to_keep:
            _, oldest_path = heapq.heappop(self._top_checkpoints)
            if oldest_path.exists() and oldest_path != checkpoint_path:
                try:
                    oldest_path.unlink()
                except OSError:
                    pass
