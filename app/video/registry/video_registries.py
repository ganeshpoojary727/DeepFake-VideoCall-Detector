"""Specialized component registries for video datasets, models, training components.

Exports pre-instantiated global registry objects matching Audio subsystem structure.
"""

from __future__ import annotations

from typing import Any, Callable, Type

from app.video.registry.base_registry import BaseRegistry


class DatasetRegistry(BaseRegistry[Any]):
    """Registry for video dataset classes."""

    def __init__(self) -> None:
        super().__init__(name="DatasetRegistry")


class ModelRegistry(BaseRegistry[Any]):
    """Registry for video model architecture classes."""

    def __init__(self) -> None:
        super().__init__(name="ModelRegistry")


class OptimizerRegistry(BaseRegistry[Any]):
    """Registry for optimizer classes and factories."""

    def __init__(self) -> None:
        super().__init__(name="OptimizerRegistry")


class SchedulerRegistry(BaseRegistry[Any]):
    """Registry for learning rate scheduler classes and factories."""

    def __init__(self) -> None:
        super().__init__(name="SchedulerRegistry")


class LossRegistry(BaseRegistry[Any]):
    """Registry for loss function modules and factories."""

    def __init__(self) -> None:
        super().__init__(name="LossRegistry")


class AugmentationRegistry(BaseRegistry[Any]):
    """Registry for spatial and temporal video data augmentations."""

    def __init__(self) -> None:
        super().__init__(name="AugmentationRegistry")


class PreprocessorRegistry(BaseRegistry[Any]):
    """Registry for video frame and face preprocessing modules."""

    def __init__(self) -> None:
        super().__init__(name="PreprocessorRegistry")


# Global instances matching framework registry convention
dataset_registry = DatasetRegistry()
model_registry = ModelRegistry()
optimizer_registry = OptimizerRegistry()
scheduler_registry = SchedulerRegistry()
loss_registry = LossRegistry()
augmentation_registry = AugmentationRegistry()
preprocessor_registry = PreprocessorRegistry()
