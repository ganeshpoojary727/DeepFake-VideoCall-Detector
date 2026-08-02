"""Specialized component registries for video datasets, models, training components.

Exports pre-instantiated global registry objects matching Audio subsystem structure.
"""

from __future__ import annotations

from app.video.registry.base_registry import BaseRegistry
from app.video.registry.dataset_registry import DatasetRegistry, dataset_registry
from app.video.registry.model_registry import ModelRegistry, model_registry
from app.video.registry.optimizer_registry import OptimizerRegistry, optimizer_registry
from app.video.registry.scheduler_registry import SchedulerRegistry, scheduler_registry
from app.video.registry.loss_registry import LossRegistry, loss_registry
from app.video.registry.augmentation_registry import AugmentationRegistry, augmentation_registry


class PreprocessorRegistry(BaseRegistry):
    """Registry for video frame and face preprocessing modules."""

    def __init__(self) -> None:
        super().__init__(name="PreprocessorRegistry")


preprocessor_registry = PreprocessorRegistry()

__all__ = [
    "BaseRegistry",
    "DatasetRegistry",
    "ModelRegistry",
    "OptimizerRegistry",
    "SchedulerRegistry",
    "LossRegistry",
    "AugmentationRegistry",
    "PreprocessorRegistry",
    "dataset_registry",
    "model_registry",
    "optimizer_registry",
    "scheduler_registry",
    "loss_registry",
    "augmentation_registry",
    "preprocessor_registry",
]
