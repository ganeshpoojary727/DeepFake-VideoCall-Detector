"""Video AI subsystem registry module exports."""

from app.video.registry.base_registry import BaseRegistry
from app.video.registry.dataset_registry import DatasetRegistry, dataset_registry
from app.video.registry.model_registry import ModelRegistry, model_registry
from app.video.registry.optimizer_registry import OptimizerRegistry, optimizer_registry
from app.video.registry.scheduler_registry import SchedulerRegistry, scheduler_registry
from app.video.registry.loss_registry import LossRegistry, loss_registry
from app.video.registry.augmentation_registry import AugmentationRegistry, augmentation_registry
from app.video.registry.video_registries import PreprocessorRegistry, preprocessor_registry

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
