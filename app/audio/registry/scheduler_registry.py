"""Scheduler registry module.

Provides SchedulerRegistry for registering and looking up PyTorch learning rate
scheduler classes (CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR, StepLR, etc.).
"""

from __future__ import annotations

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)

from app.audio.registry.base_registry import BaseRegistry


class SchedulerRegistry(BaseRegistry[LRScheduler]):
    """Registry for PyTorch learning rate schedulers."""

    def __init__(self) -> None:
        super().__init__(name="SchedulerRegistry")
        self.register("cosine", CosineAnnealingLR)
        self.register("plateau", ReduceLROnPlateau)
        self.register("step", StepLR)
        self.register("onecycle", OneCycleLR)


# Default global instance for schedulers
scheduler_registry = SchedulerRegistry()
