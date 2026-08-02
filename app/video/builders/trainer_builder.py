"""Fluent VideoTrainerBuilder module for assembling training pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from app.video.augmentation.augmentation_pipeline import VideoAugmentationPipeline
from app.video.builders.augmentation_builder import AugmentationBuilder
from app.video.builders.dataset_builder import DatasetBuilder
from app.video.builders.model_builder import VideoModelBuilder
from app.video.configs.augmentation_config import AugmentationConfig
from app.video.configs.dataset_config import DatasetConfig
from app.video.configs.model_config import ModelConfig
from app.video.configs.training_config import VideoTrainingConfig
from app.video.training.loss_factory import LossFactory
from app.video.training.optimizer_factory import OptimizerFactory
from app.video.training.scheduler_factory import SchedulerFactory
from app.video.training.trainer import ProductionVideoTrainer


class VideoTrainerBuilder:
    """Fluent builder for building ProductionVideoTrainer instances."""

    def __init__(self) -> None:
        self.model_config = ModelConfig()
        self.training_config = VideoTrainingConfig()
        self.dataset_config = DatasetConfig()
        self.augmentation_config = AugmentationConfig()

        self._model: Optional[nn.Module] = None
        self._train_loader: Optional[DataLoader] = None
        self._val_loader: Optional[DataLoader] = None

    def with_model_config(self, config: ModelConfig) -> VideoTrainerBuilder:
        """Set model configuration."""
        self.model_config = config
        return self

    def with_training_config(self, config: VideoTrainingConfig) -> VideoTrainerBuilder:
        """Set training configuration."""
        self.training_config = config
        return self

    def with_dataset_config(self, config: DatasetConfig) -> VideoTrainerBuilder:
        """Set dataset configuration."""
        self.dataset_config = config
        return self

    def with_model(self, model: nn.Module) -> VideoTrainerBuilder:
        """Explicitly set PyTorch model instance."""
        self._model = model
        return self

    def with_dataloaders(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None
    ) -> VideoTrainerBuilder:
        """Explicitly set training and validation dataloaders."""
        self._train_loader = train_loader
        self._val_loader = val_loader
        return self

    def build(self) -> ProductionVideoTrainer:
        """Assemble all components and return a ProductionVideoTrainer instance."""
        model = self._model or VideoModelBuilder().build(self.model_config)

        optimizer = OptimizerFactory.create_optimizer(model, config=self.training_config)
        loss_fn = LossFactory.create_loss(config=self.training_config)
        scheduler = SchedulerFactory.create_scheduler(optimizer, config=self.training_config)

        return ProductionVideoTrainer(
            model=model,
            train_loader=self._train_loader,
            val_loader=self._val_loader,
            config=self.training_config,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
        )


# Class alias
TrainerBuilder = VideoTrainerBuilder
