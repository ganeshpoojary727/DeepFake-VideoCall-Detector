"""Model registry module for video AI subsystem."""

from __future__ import annotations

from typing import Any
from app.video.registry.base_registry import BaseRegistry


class ModelRegistry(BaseRegistry[Any]):
    """Registry for video model architecture classes."""

    def __init__(self) -> None:
        super().__init__(name="ModelRegistry")


# Global model registry instance
model_registry = ModelRegistry()
