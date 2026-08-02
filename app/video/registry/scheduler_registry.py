"""Scheduler registry module for video AI subsystem."""

from __future__ import annotations

from typing import Any
from app.video.registry.base_registry import BaseRegistry


class SchedulerRegistry(BaseRegistry[Any]):
    """Registry for learning rate scheduler classes and factories."""

    def __init__(self) -> None:
        super().__init__(name="SchedulerRegistry")


# Global scheduler registry instance
scheduler_registry = SchedulerRegistry()
