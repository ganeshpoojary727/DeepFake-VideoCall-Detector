"""Loss registry module for video AI subsystem."""

from __future__ import annotations

from typing import Any
from app.video.registry.base_registry import BaseRegistry


class LossRegistry(BaseRegistry[Any]):
    """Registry for loss function modules and factories."""

    def __init__(self) -> None:
        super().__init__(name="LossRegistry")


# Global loss registry instance
loss_registry = LossRegistry()
