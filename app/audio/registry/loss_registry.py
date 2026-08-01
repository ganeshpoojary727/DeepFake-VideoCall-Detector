"""Loss function registry module.

Provides LossRegistry for registering, looking up, and instantiating loss functions
(CrossEntropyLoss, FocalLoss, OCSoftmax, CenterLoss, etc.).
"""

from __future__ import annotations

import torch.nn as nn

from app.audio.registry.base_registry import BaseRegistry


class LossRegistry(BaseRegistry[nn.Module]):
    """Registry for training loss functions."""

    def __init__(self) -> None:
        super().__init__(name="LossRegistry")
        self.register("cross_entropy", nn.CrossEntropyLoss)
        self.register("bce", nn.BCELoss)
        self.register("bce_with_logits", nn.BCEWithLogitsLoss)
        self.register("mse", nn.MSELoss)


# Default global instance for loss functions
loss_registry = LossRegistry()
