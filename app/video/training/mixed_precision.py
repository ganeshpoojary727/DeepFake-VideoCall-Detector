"""Mixed precision training wrapper module for Automatic Mixed Precision (AMP)."""

from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn


class MixedPrecisionHandler:
    """Wrapper for PyTorch Automatic Mixed Precision (AMP) GradScaler and autocast context."""

    def __init__(self, enabled: bool = True, device: str = "cuda") -> None:
        self.enabled = enabled and torch.cuda.is_available() and device.startswith("cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.enabled)

    def autocast(self) -> Any:
        """Get autocast context manager."""
        return torch.amp.autocast("cuda", enabled=self.enabled)

    def scale_and_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        clip_norm: float = 1.0,
        model: torch.nn.Module | None = None,
    ) -> None:
        """Backward pass with scaled loss, gradient clipping, and optimizer step."""
        if self.enabled:
            self.scaler.scale(loss).backward()
            if model is not None and clip_norm > 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if model is not None and clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
