"""Loss function factory module."""

from __future__ import annotations

from typing import Any, Dict, Type
import torch
import torch.nn as nn

from app.video.exceptions.video_exceptions import ConfigurationError
from app.video.registry.video_registries import loss_registry


class FocalLoss(nn.Module):
    """Focal Loss implementation for handling class imbalance in video deepfakes."""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class LossFactory:
    """Factory for creating loss function criterion instances."""

    _mapping: Dict[str, Type[nn.Module]] = {
        "cross_entropy": nn.CrossEntropyLoss,
        "bce": nn.BCEWithLogitsLoss,
        "focal": FocalLoss,
    }

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> nn.Module:
        """Create loss criterion by name.

        Args:
            name: Loss function key ("cross_entropy", "bce", "focal").

        Returns:
            nn.Module: Instantiated PyTorch loss criterion.
        """
        key = name.lower().strip()
        if key in cls._mapping:
            loss_cls = cls._mapping[key]
        else:
            try:
                loss_cls = loss_registry.get(key)
            except Exception as err:
                raise ConfigurationError(f"Unsupported loss name '{name}'") from err

        return loss_cls(**kwargs)


# Register defaults in global registry
for loss_key, loss_class in LossFactory._mapping.items():
    loss_registry.register(loss_key, loss_class, overwrite=True)
