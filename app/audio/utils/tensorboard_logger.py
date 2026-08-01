"""TensorBoard logging integration module."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch


class TensorBoardLogger:
    """Handles writing training metrics, validation curves, LR, and histograms to TensorBoard."""

    def __init__(self, log_dir: str | Path = "logs/audio/tensorboard") -> None:
        self.log_dir = Path(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = None

        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        except ImportError:
            pass

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar metric."""
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "train") -> None:
        """Log metric dictionary values."""
        if self.writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"{prefix}/{k}", v, step)

    def log_model_histograms(self, model: torch.nn.Module, step: int) -> None:
        """Log parameter and gradient histograms."""
        if self.writer is not None:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(f"params/{name}", param, step)
                    if param.grad is not None:
                        self.writer.add_histogram(f"grads/{name}", param.grad, step)

    def close(self) -> None:
        """Close SummaryWriter connection."""
        if self.writer is not None:
            self.writer.close()
