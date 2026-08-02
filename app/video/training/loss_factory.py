"""Loss function factory module supporting CrossEntropy, Focal, Weighted, and Label Smoothing loss."""

from __future__ import annotations

from typing import Any, Dict, Optional, Type
import torch
import torch.nn as nn

from app.video.configs.training_config import VideoTrainingConfig
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


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross Entropy Loss for imbalanced class distributions."""

    def __init__(self, weights: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weights)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted cross entropy loss."""
        return self.ce(inputs, targets)


class LossFactory:
    """Factory for creating loss function criterion instances."""

    _mapping: Dict[str, Type[nn.Module]] = {
        "cross_entropy": nn.CrossEntropyLoss,
        "bce": nn.BCEWithLogitsLoss,
        "focal": FocalLoss,
        "weighted_ce": WeightedCrossEntropyLoss,
    }

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> nn.Module:
        """Create loss criterion by name."""
        key = name.lower().strip()
        if key in cls._mapping:
            loss_cls = cls._mapping[key]
        else:
            try:
                loss_cls = loss_registry.get(key)
            except Exception as err:
                raise ConfigurationError(f"Unsupported loss name '{name}'") from err

        return loss_cls(**kwargs)

    @classmethod
    def create_loss(
        cls,
        config: Optional[VideoTrainingConfig] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> nn.Module:
        """Create loss criterion from VideoTrainingConfig or name."""
        loss_name = name or (config.loss_name if config else "cross_entropy")
        if loss_name == "cross_entropy" and config and hasattr(config, "label_smoothing") and config.label_smoothing > 0:
            kwargs["label_smoothing"] = config.label_smoothing
        return cls.create(name=loss_name, **kwargs)


# Register defaults in global registry
for loss_key, loss_class in LossFactory._mapping.items():
    loss_registry.register(loss_key, loss_class, overwrite=True)
