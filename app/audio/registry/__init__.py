"""Registry pattern subpackage."""

from app.audio.registry.base_registry import BaseRegistry
from app.audio.registry.dataset_registry import DatasetRegistry, dataset_registry
from app.audio.registry.loss_registry import LossRegistry, loss_registry
from app.audio.registry.model_registry import ModelRegistry, model_registry
from app.audio.registry.optimizer_registry import OptimizerRegistry, optimizer_registry
from app.audio.registry.scheduler_registry import SchedulerRegistry, scheduler_registry

__all__ = [
    "BaseRegistry",
    "ModelRegistry",
    "model_registry",
    "DatasetRegistry",
    "dataset_registry",
    "LossRegistry",
    "loss_registry",
    "OptimizerRegistry",
    "optimizer_registry",
    "SchedulerRegistry",
    "scheduler_registry",
]
