"""Learning rate scheduler factory module."""

from __future__ import annotations

from typing import Any, Dict, Type
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    ReduceLROnPlateau,
    StepLR,
)

from app.video.exceptions.video_exceptions import ConfigurationError
from app.video.registry.video_registries import scheduler_registry


class SchedulerFactory:
    """Factory for creating learning rate schedulers."""

    _mapping: Dict[str, Any] = {
        "cosine": CosineAnnealingLR,
        "step": StepLR,
        "plateau": ReduceLROnPlateau,
    }

    @classmethod
    def create(
        cls,
        name: str,
        optimizer: Optimizer,
        epochs: int = 50,
        **kwargs: Any,
    ) -> Any:
        """Create learning rate scheduler instance.

        Args:
            name: Scheduler key name ("cosine", "step", "plateau").
            optimizer: Target optimizer instance.
            epochs: Total training epochs.

        Returns:
            Any: Instantiated learning rate scheduler.
        """
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
        else:
            try:
                sched_cls = scheduler_registry.get(key)
                return sched_cls(optimizer, **kwargs)
            except Exception as err:
                raise ConfigurationError(f"Unsupported scheduler name '{name}'") from err


# Register defaults in global registry
for sched_key, sched_class in SchedulerFactory._mapping.items():
    scheduler_registry.register(sched_key, sched_class, overwrite=True)
