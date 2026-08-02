"""Video training components builder module."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from app.video.configs.model_config import ModelConfig
from app.video.configs.training_config import VideoTrainingConfig
from app.video.builders.model_builder import VideoModelBuilder
from app.video.training.loss_factory import LossFactory
from app.video.training.optimizer_factory import OptimizerFactory
from app.video.training.scheduler_factory import SchedulerFactory
from app.video.training.trainer import VideoTrainer


class TrainingBuilder:
    """Builder for assembling video model training components into VideoTrainer."""

    def __init__(self) -> None:
        self._model_builder = VideoModelBuilder()

    def build_trainer(
        self,
        training_config: Optional[VideoTrainingConfig] = None,
        model_config: Optional[ModelConfig] = None,
        model: Optional[nn.Module] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
    ) -> VideoTrainer:
        """Construct VideoTrainer with configured model, optimizer, loss, and scheduler.

        Args:
            training_config: Training configuration object.
            model_config: Model configuration object.
            model: Optional pre-constructed model instance.
            train_loader: Optional training dataloader.
            val_loader: Optional validation dataloader.

        Returns:
            VideoTrainer: Configured training engine.
        """
        t_cfg = training_config or VideoTrainingConfig()
        m_cfg = model_config or ModelConfig()

        if model is None:
            model = self._model_builder.build(m_cfg)

        optimizer = OptimizerFactory.create_optimizer(model, config=t_cfg)
        loss_fn = LossFactory.create_loss(config=t_cfg)
        scheduler = SchedulerFactory.create_scheduler(optimizer, config=t_cfg)

        return VideoTrainer(
            model=model,
            config=t_cfg,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
        )
