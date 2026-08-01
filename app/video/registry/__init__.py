"""Video component registries package."""

from app.video.registry.base_registry import BaseRegistry
from app.video.registry.video_registries import (
    AugmentationRegistry,
    DatasetRegistry,
    LossRegistry,
    ModelRegistry,
    OptimizerRegistry,
    PreprocessorRegistry,
    SchedulerRegistry,
    augmentation_registry,
    dataset_registry,
    loss_registry,
    model_registry,
    optimizer_registry,
    preprocessor_registry,
    scheduler_registry,
)

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
