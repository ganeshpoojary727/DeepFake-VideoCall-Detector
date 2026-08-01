"""Training callbacks and callback handler module."""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List


class BaseCallback(ABC):
    """Abstract base class for training execution hooks."""

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at start of training."""
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at start of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at end of each epoch."""
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at end of training loop."""
        pass


class CallbackHandler:
    """Dispatches event triggers to registered callbacks list."""

    def __init__(self, callbacks: Optional[List[BaseCallback]] = None) -> None:
        self.callbacks = callbacks or []

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Trigger start of training."""
        for cb in self.callbacks:
            cb.on_train_begin(logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Trigger start of epoch."""
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Trigger end of epoch."""
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Trigger end of training."""
        for cb in self.callbacks:
            cb.on_train_end(logs)


class LoggingCallback(BaseCallback):
    """Callback for logging training metrics at epoch completion."""

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log epoch metric details."""
        if logs:
            loss = logs.get("val_loss", logs.get("train_loss", 0.0))
            # Epoch log placeholder


class ModelCheckpointCallback(BaseCallback):
    """Callback for saving checkpoint on epoch end."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Save checkpoint at epoch end."""
        pass
