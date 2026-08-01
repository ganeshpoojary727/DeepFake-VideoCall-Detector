"""Learning rate scheduler factory module for audio models."""

from __future__ import annotations

import math
from typing import Any, Optional
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)

from app.audio.configs.training_config import AudioTrainingConfig
from app.audio.registry.scheduler_registry import scheduler_registry


class WarmupCosineLR(LambdaLR):
    """Linear warmup followed by cosine annealing decay scheduler."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int = 5,
        max_epochs: int = 50,
        min_lr_ratio: float = 1e-3,
    ) -> None:
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = float(epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

        super().__init__(optimizer, lr_lambda=lr_lambda)


class SchedulerFactory:
    """Factory for instantiating PyTorch learning rate schedulers."""

    def __init__(self, config: AudioTrainingConfig) -> None:
        self.config = config

    def create_scheduler(
        self,
        optimizer: Optimizer,
        steps_per_epoch: Optional[int] = None,
    ) -> Any:
        """Create PyTorch learning rate scheduler."""
        name = self.config.scheduler_name.lower().strip()

        if name in ("cosine", "cosineannealing"):
            return CosineAnnealingLR(optimizer, T_max=self.config.epochs, eta_min=1e-6)
        elif name in ("onecycle", "onecyclelr"):
            steps = steps_per_epoch or 100
            return OneCycleLR(
                optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.epochs,
                steps_per_epoch=steps,
            )
        elif name in ("warmup_cosine", "warmupcosine"):
            return WarmupCosineLR(optimizer, warmup_epochs=5, max_epochs=self.config.epochs)
        elif name in ("plateau", "reducelronplateau"):
            return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        elif name in ("step", "steplr"):
            return StepLR(optimizer, step_size=5, gamma=0.5)
        else:
            try:
                sched_cls = scheduler_registry.get(name)
                return sched_cls(optimizer)
            except Exception as err:
                raise ValueError(f"Unsupported scheduler type: '{name}'. Supported: cosine, onecycle, warmup_cosine, plateau.") from err


# Register defaults in scheduler_registry
scheduler_registry.register("cosine", CosineAnnealingLR, overwrite=True)
scheduler_registry.register("onecycle", OneCycleLR, overwrite=True)
scheduler_registry.register("warmup_cosine", WarmupCosineLR, overwrite=True)
scheduler_registry.register("plateau", ReduceLROnPlateau, overwrite=True)
