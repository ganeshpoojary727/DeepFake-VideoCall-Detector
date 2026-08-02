"""Optimizer registry module for video AI subsystem."""

from __future__ import annotations

from typing import Any
from app.video.registry.base_registry import BaseRegistry


class OptimizerRegistry(BaseRegistry[Any]):
    """Registry for optimizer classes and factories."""

    def __init__(self) -> None:
        super().__init__(name="OptimizerRegistry")


# Global optimizer registry instance
optimizer_registry = OptimizerRegistry()
