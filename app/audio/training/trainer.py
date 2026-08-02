"""Production Audio Trainer orchestration engine for AASIST model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.audio.configs.training_config import AudioTrainingConfig
from app.audio.training.checkpoint import CheckpointManager
from app.audio.training.ema import EMAModel
from app.audio.training.loss_factory import LossFactory
from app.audio.training.optimizer import OptimizerFactory
from app.audio.training.scheduler import SchedulerFactory
from app.audio.training.validator import ValidationEngine
from app.audio.utils.audio_logger import get_audio_logger
from app.audio.utils.tensorboard_logger import TensorBoardLogger

logger = get_audio_logger("training.trainer")


class ProductionAudioTrainer:
    """Production training engine for AASIST audio deepfake detector."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[AudioTrainingConfig] = None,
        **kwargs: Any,
    ) -> None:
        self.config = config or AudioTrainingConfig()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader or kwargs.get("validation_loader")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Factories
        self.optimizer = OptimizerFactory(self.config).create_optimizer(self.model)
        self.scheduler = SchedulerFactory(self.config).create_scheduler(
            self.optimizer, steps_per_epoch=len(train_loader)
        )
        self.criterion = LossFactory(self.config).create_loss()

        # Components
        self.validator = ValidationEngine(self.model, device=str(self.device))
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir,
            max_to_keep=3,
        )
        self.tb_logger = TensorBoardLogger(log_dir=self.config.tensorboard_dir)
        self.ema = EMAModel(self.model, decay=self.config.ema_decay) if self.config.use_ema else None

        # AMP
        self.use_amp = self.config.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        self.best_val_loss = float("inf")
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": [], "val_eer": []}

    def train_epoch(self, epoch: int) -> float:
        """Execute single epoch iteration with gradient accumulation and AMP."""
        self.model.train()
        running_loss = 0.0
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(self.train_loader):
            if isinstance(batch, dict):
                x = batch["tensor"].to(self.device, non_blocking=True)
                y = batch["label"].to(self.device, non_blocking=True)
            else:
                x, y = batch[0].to(self.device), batch[1].to(self.device)

            if self.use_amp and self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = self.model(x)
                    loss = self.criterion(logits, y) / self.config.grad_accum_steps
                self.scaler.scale(loss).backward()

                if (step + 1) % self.config.grad_accum_steps == 0:
                    if self.config.gradient_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                logits = self.model(x)
                loss = self.criterion(logits, y) / self.config.grad_accum_steps
                loss.backward()

                if (step + 1) % self.config.grad_accum_steps == 0:
                    if self.config.gradient_clip_norm > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            if self.ema is not None:
                self.ema.update(self.model)

            running_loss += loss.item() * self.config.grad_accum_steps

        epoch_loss = running_loss / len(self.train_loader)
        return float(epoch_loss)

    def train(self) -> Dict[str, Any]:
        """Execute complete multi-epoch training pipeline."""
        logger.info("Starting AASIST training — %d epochs, device=%s", self.config.epochs, self.device)

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(epoch)
            self.history["train_loss"].append(train_loss)

            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self.validator.evaluate(self.val_loader, self.criterion)
                val_loss = val_metrics.get("val_loss", 0.0)
                val_acc = val_metrics.get("accuracy", 0.0)
                val_eer = val_metrics.get("eer", 0.0)

                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                self.history["val_eer"].append(val_eer)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.checkpoint_manager.save_best(self.model, epoch, val_metrics)

            if self.scheduler is not None:
                if hasattr(self.scheduler, "step"):
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics.get("val_loss", train_loss))
                    else:
                        self.scheduler.step()

            # Logging & Checkpointing
            lr = self.optimizer.param_groups[0]["lr"]
            self.tb_logger.log_scalar("train/loss", train_loss, epoch)
            self.tb_logger.log_scalar("train/lr", lr, epoch)
            if val_metrics:
                self.tb_logger.log_metrics(val_metrics, epoch, prefix="val")

            self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metric_value=train_loss,
                history=self.history,
            )

        self.tb_logger.close()
        return self.history

    def fit(self, epochs: Optional[int] = None) -> Dict[str, Any]:
        """Alias method for training loop execution."""
        if epochs is not None:
            self.config.epochs = epochs
        return self.train()


# Backward compatibility alias
Trainer = ProductionAudioTrainer