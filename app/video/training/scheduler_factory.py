"""Learning rate scheduler factory module supporting Cosine, Plateau, Step, OneCycle, and Warmup."""

from __future__ import annotations

from typing import Any, Dict, Optional, Type
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)

from app.video.configs.training_config import VideoTrainingConfig
from app.video.exceptions.video_exceptions import ConfigurationError
from app.video.registry.video_registries import scheduler_registry


class LinearWarmupLR(LRScheduler):
    """Linear warmup learning rate scheduler wrapper."""

    def __init__(self, optimizer: Optimizer, warmup_epochs: int = 5, max_epochs: int = 50) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        """Compute learning rate with initial linear warmup scaling."""
        if self.last_epoch < self.warmup_epochs:
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        return self.base_lrs


class SchedulerFactory:
    """Factory for creating learning rate schedulers."""

    _mapping: Dict[str, Any] = {
        "cosine": CosineAnnealingLR,
        "step": StepLR,
        "plateau": ReduceLROnPlateau,
        "onecycle": OneCycleLR,
        "warmup": LinearWarmupLR,
    }

    @classmethod
    def create(
        cls,
        name: str,
        optimizer: Optimizer,
        epochs: int = 50,
        **kwargs: Any,
    ) -> Any:
        """Create learning rate scheduler instance."""
        key = name.lower().strip()
        if key == "cosine":
            T_max = kwargs.get("T_max", epochs)
            return CosineAnnealingLR(optimizer, T_max=T_max)
        elif key == "step":
            step_size = kwargs.get("step_size", 10)
            gamma = kwargs.get("gamma", 0.1)
            return StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif key == "plateau":
            patience = kwargs.get("patience", 5)
            return ReduceLROnPlateau(optimizer, patience=patience)
        elif key == "onecycle":
            max_lr = kwargs.get("max_lr", 1e-3)
            total_steps = kwargs.get("total_steps", epochs)
            return OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps)
        elif key == "warmup":
            warmup_epochs = kwargs.get("warmup_epochs", 5)
            return LinearWarmupLR(optimizer, warmup_epochs=warmup_epochs, max_epochs=epochs)
        else:
            try:
                sched_cls = scheduler_registry.get(key)
                return sched_cls(optimizer, **kwargs)
            except Exception as err:
                raise ConfigurationError(f"Unsupported scheduler name '{name}'") from err

    @classmethod
    def create_scheduler(
        cls,
        optimizer: Optimizer,
        config: Optional[VideoTrainingConfig] = None,
    ) -> Any:
        """Create scheduler from optimizer and VideoTrainingConfig."""
        cfg = config or VideoTrainingConfig()
        return cls.create(
            name=cfg.scheduler_name,
            optimizer=optimizer,
            epochs=cfg.epochs,
        )


# Register defaults in global registry
for sched_key, sched_class in SchedulerFactory._mapping.items():
    scheduler_registry.register(sched_key, sched_class, overwrite=True)
