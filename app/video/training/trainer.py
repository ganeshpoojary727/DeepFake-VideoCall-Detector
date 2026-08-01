"""Video model trainer orchestration module."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.video.configs.training_config import VideoTrainingConfig
from app.video.exceptions.video_exceptions import TrainingError
from app.video.training.callbacks import CallbackHandler
from app.video.training.checkpoint_manager import CheckpointManager
from app.video.training.early_stopping import EarlyStopping
from app.video.training.loss_factory import LossFactory
from app.video.training.metrics import VideoMetricsCalculator
from app.video.training.optimizer_factory import OptimizerFactory
from app.video.training.scheduler_factory import SchedulerFactory
from app.video.utils.device import get_device


class Trainer:
    """Orchestrates video deepfake detector model training, validation, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        config: Optional[VideoTrainingConfig] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> None:
        self.config = config or VideoTrainingConfig()
        self.model = model
        self.device = get_device(self.config.device)
        self.model.to(self.device)

        self.optimizer = OptimizerFactory.create(
            name=self.config.optimizer_name,
            params=self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = SchedulerFactory.create(
            name=self.config.scheduler_name,
            optimizer=self.optimizer,
            epochs=self.config.epochs,
        )

        self.criterion = LossFactory.create(name=self.config.loss_name)

        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
        )

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir,
            save_top_k=self.config.save_top_k,
        )

        self.callback_handler = CallbackHandler(callbacks=callbacks)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run single training epoch step."""
        self.model.train()
        total_loss = 0.0
        count = 0

        for batch in dataloader:
            if isinstance(batch, dict):
                x = batch["tensor"].to(self.device)
                y = batch["label"].to(self.device)
            elif hasattr(batch, "tensor"):
                x = batch.tensor.to(self.device)
                y = torch.tensor([batch.label], device=self.device)
            elif isinstance(batch, (tuple, list)):
                x, y = batch[0].to(self.device), batch[1].to(self.device)
            else:
                continue

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()

            if self.config.gradient_clip_val > 0.0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)

            self.optimizer.step()
            total_loss += loss.item() * x.size(0)
            count += x.size(0)

        return total_loss / count if count > 0 else 0.0

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run validation step and compute metrics."""
        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    x = batch["tensor"].to(self.device)
                    y = batch["label"].to(self.device)
                elif hasattr(batch, "tensor"):
                    x = batch.tensor.to(self.device)
                    y = torch.tensor([batch.label], device=self.device)
                elif isinstance(batch, (tuple, list)):
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    continue

                logits = self.model(x)
                loss = self.criterion(logits, y)
                total_loss += loss.item() * x.size(0)
                all_logits.append(logits.cpu())
                all_labels.append(y.cpu())

        val_loss = total_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0.0

        if all_logits:
            cat_logits = torch.cat(all_logits, dim=0)
            cat_labels = torch.cat(all_labels, dim=0)
            metrics = VideoMetricsCalculator.compute_all(cat_logits, cat_labels)
        else:
            metrics = {"accuracy": 0.0, "f1": 0.0}

        metrics["val_loss"] = val_loss
        return metrics
