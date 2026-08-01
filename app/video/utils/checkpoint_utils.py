"""Model checkpoint saving and loading utility functions."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional
import torch


def save_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Save PyTorch model weights and metadata checkpoint to disk.

    Args:
        filepath: Destination file path.
        model: PyTorch module.
        optimizer: PyTorch optimizer instance.
        scheduler: Learning rate scheduler.
        epoch: Training epoch count.
        metrics: Computed validation metrics dict.
        extra_meta: Additional metadata dictionary.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
        "extra_meta": extra_meta or {},
    }
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        state["scheduler_state"] = scheduler.state_dict()
    torch.save(state, filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load PyTorch model state and metadata from checkpoint.

    Args:
        filepath: Checkpoint path.
        model: PyTorch module.
        optimizer: Optimizer instance to restore.
        scheduler: Scheduler instance to restore.
        device: Target torch device.

    Returns:
        Dict[str, Any]: Restored checkpoint payload dictionary.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    payload = torch.load(filepath, map_location=device)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and "scheduler_state" in payload and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(payload["scheduler_state"])
    return payload


def inspect_checkpoint(filepath: str) -> Dict[str, Any]:
    """Inspect metadata stored inside a checkpoint file without loading model.

    Args:
        filepath: Checkpoint path.

    Returns:
        Dict[str, Any]: Summary dictionary containing epoch and metrics.
    """
    payload = torch.load(filepath, map_location="cpu")
    return {
        "epoch": payload.get("epoch", 0),
        "metrics": payload.get("metrics", {}),
        "extra_meta": payload.get("extra_meta", {}),
        "keys": list(payload.keys()),
    }
