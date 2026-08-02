"""Dataset registry module for video AI subsystem."""

from __future__ import annotations

from typing import Any
from app.video.registry.base_registry import BaseRegistry


class DatasetRegistry(BaseRegistry[Any]):
    """Registry for video dataset classes."""

    def __init__(self) -> None:
        super().__init__(name="DatasetRegistry")


# Global dataset registry instance
dataset_registry = DatasetRegistry()
