"""Augmentation registry module for video AI subsystem."""

from __future__ import annotations

from typing import Any
from app.video.registry.base_registry import BaseRegistry


class AugmentationRegistry(BaseRegistry[Any]):
    """Registry for spatial and temporal video data augmentations."""

    def __init__(self) -> None:
        super().__init__(name="AugmentationRegistry")


# Global augmentation registry instance
augmentation_registry = AugmentationRegistry()
