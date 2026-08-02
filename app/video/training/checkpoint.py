"""Model checkpoint saving, loading, and resume utility module for video AI models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn

from app.video.exceptions.video_exceptions import ModelCheckpointError


class CheckpointManager:
    """Manages serializing model state, optimizer, scheduler, and training progress to disk."""

    def __init__(self, checkpoint_dir: str | Path = "trained_models/video", max_to_keep: int = 5) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_to_keep = max_to_keep
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        metrics: Optional[Dict[str, float]] = None,
        scheduler: Optional[Any] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """Save model state checkpoint to file.

        Args:
            model: PyTorch neural network model.
            optimizer: PyTorch optimizer instance.
            epoch: Current epoch index.
            loss: Validation or training loss.
            metrics: Optional metrics dictionary.
            scheduler: Optional learning rate scheduler.
            filename: Optional custom filename.

        Returns:
            Path: Written checkpoint destination path.
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:03d}_loss_{loss:.4f}.pt"
        filepath = self.checkpoint_dir / filename

        state: Dict[str, Any] = {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics or {},
        }
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            state["scheduler_state_dict"] = scheduler.state_dict()

        try:
            torch.save(state, filepath)
            return filepath
        except Exception as e:
            raise ModelCheckpointError(f"Failed to save checkpoint to {filepath}: {e}") from e

    def load_checkpoint(
        self,
        filepath: str | Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[int, float, Dict[str, float]]:
        """Load checkpoint state into model and optimizer for resume training.

        Args:
            filepath: Checkpoint file path.
            model: Target PyTorch model.
            optimizer: Optional target optimizer.
            scheduler: Optional target scheduler.
            device: Computing device.

        Returns:
            Tuple[int, float, Dict[str, float]]: (epoch, loss, metrics).
        """
        path = Path(filepath)
        if not path.exists():
            raise ModelCheckpointError(f"Checkpoint file not found: {path}")

        try:
            checkpoint = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            epoch = checkpoint.get("epoch", 0)
            loss = checkpoint.get("loss", 0.0)
            metrics = checkpoint.get("metrics", {})
            return epoch, loss, metrics
        except Exception as e:
            raise ModelCheckpointError(f"Failed to load checkpoint from {path}: {e}") from e

    def find_latest_checkpoint(self) -> Optional[Path]:
        """Find most recently saved checkpoint file in directory."""
        checkpoints = sorted(self.checkpoint_dir.glob("*.pt"))
        return checkpoints[-1] if checkpoints else None
