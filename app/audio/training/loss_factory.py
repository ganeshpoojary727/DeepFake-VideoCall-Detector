"""Loss function factory module for audio deepfake classification."""

from __future__ import annotations

from typing import Any, Optional
import torch
import torch.nn as nn

from app.audio.configs.training_config import AudioTrainingConfig
from app.audio.registry.loss_registry import loss_registry


class FocalLoss(nn.Module):
    """Focal Loss implementation for handling class imbalance in audio deepfakes."""

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


class ClassBalancedLoss(nn.Module):
    """Class-balanced loss weighted by inverse class frequency."""

    def __init__(self, num_bonafide: int = 2580, num_spoof: int = 22800) -> None:
        super().__init__()
        total = num_bonafide + num_spoof
        w_bonafide = total / (2 * num_bonafide)
        w_spoof = total / (2 * num_spoof)
        weights = torch.tensor([w_bonafide, w_spoof], dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=weights)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute class-balanced weighted cross entropy."""
        weights = self.ce.weight.to(inputs.device)
        return nn.functional.cross_entropy(inputs, targets, weight=weights)


class LossFactory:
    """Factory for creating loss function criteria."""

    def __init__(self, config: AudioTrainingConfig) -> None:
        self.config = config

    def create_loss(self) -> nn.Module:
        """Create PyTorch loss module."""
        name = self.config.loss_name.lower().strip()

        if name == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif name in ("label_smoothing", "labelsmoothing"):
            return nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        elif name == "focal":
            return FocalLoss(alpha=self.config.focal_alpha, gamma=self.config.focal_gamma)
        elif name in ("class_balanced", "classbalanced"):
            return ClassBalancedLoss()
        else:
            try:
                loss_cls = loss_registry.get(name)
                return loss_cls()
            except Exception as err:
                raise ValueError(f"Unsupported loss type: '{name}'. Supported: cross_entropy, label_smoothing, focal, class_balanced.") from err


# Register default loss functions in loss_registry
loss_registry.register("cross_entropy", nn.CrossEntropyLoss, overwrite=True)
loss_registry.register("focal", FocalLoss, overwrite=True)
loss_registry.register("class_balanced", ClassBalancedLoss, overwrite=True)
