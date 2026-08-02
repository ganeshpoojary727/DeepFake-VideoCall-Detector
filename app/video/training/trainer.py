"""Production Video Trainer orchestration engine mirroring Audio subsystem."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from app.video.configs.training_config import VideoTrainingConfig
from app.video.core.base_trainer import BaseTrainer
from app.video.datasets.dataloader import video_collate_fn
from app.video.evaluation.metrics import EvaluationMetrics
from app.video.exceptions.video_exceptions import TrainingError
from app.video.training.checkpoint_manager import CheckpointManager
from app.video.training.early_stopping import EarlyStopping
from app.video.training.loss_factory import LossFactory
from app.video.training.metrics import VideoMetricsCalculator
from app.video.training.mixed_precision import MixedPrecisionHandler
from app.video.training.optimizer_factory import OptimizerFactory
from app.video.training.scheduler_factory import SchedulerFactory

logger = logging.getLogger(__name__)


class ProductionVideoTrainer(BaseTrainer):
    """Production training engine for video deepfake detector models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        config: Optional[VideoTrainingConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[nn.Module] = None,
        scheduler: Optional[Any] = None,
    ) -> None:
        self.config = config or VideoTrainingConfig()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device(self.config.device if torch.cuda.is_available() and "cuda" in self.config.device else "cpu")
        self.model.to(self.device)

        self.optimizer = optimizer or OptimizerFactory.create_optimizer(self.model, config=self.config)
        self.loss_fn = loss_fn or LossFactory.create_loss(config=self.config)
        self.scheduler = scheduler or SchedulerFactory.create_scheduler(self.optimizer, config=self.config)

        self.amp_handler = MixedPrecisionHandler(enabled=self.config.use_amp, device=str(self.device))
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
        )
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir,
            save_top_k=self.config.save_top_k,
        )

        self.best_val_loss = float("inf")
        self.best_accuracy = 0.0
        self.best_f1 = 0.0

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "val_auc": [],
        }

    def train_epoch(self, dataloader_or_epoch: Optional[Union[DataLoader, int]] = None) -> float:
        """Execute single training epoch iteration with AMP, gradient accumulation, and clipping."""
        loader = self.train_loader
        if isinstance(dataloader_or_epoch, DataLoader):
            loader = dataloader_or_epoch

        if loader is None:
            raise TrainingError("train_loader is not set for trainer.")

        self.model.train()
        running_loss = 0.0
        total_samples = 0
        accum_steps = self.config.gradient_accumulation_steps

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            if isinstance(batch, dict):
                x = batch["tensor"].to(self.device, non_blocking=True)
                y = batch["label"].to(self.device, non_blocking=True)
            elif hasattr(batch, "tensor"):
                x = batch.tensor.to(self.device)
                y = torch.tensor([batch.label], device=self.device)
            elif isinstance(batch, (tuple, list)):
                x, y = batch[0].to(self.device), batch[1].to(self.device)
            else:
                continue

            with self.amp_handler.autocast():
                logits = self.model(x)
                loss = self.loss_fn(logits, y) / accum_steps

            self.amp_handler.scale_and_step(
                loss=loss,
                optimizer=self.optimizer,
                clip_norm=self.config.gradient_clip_norm if (step + 1) % accum_steps == 0 else 0.0,
                model=self.model,
            )

            if (step + 1) % accum_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)

            batch_size = x.size(0)
            running_loss += (loss.item() * accum_steps) * batch_size
            total_samples += batch_size

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
        return float(epoch_loss)

    def validate(
        self,
        dataloader_or_dataset: Optional[Union[DataLoader, Dataset]] = None,
        batch_size: int = 4,
    ) -> Dict[str, float]:
        """Execute evaluation pass over validation dataset split."""
        loader = self.val_loader

        if isinstance(dataloader_or_dataset, DataLoader):
            loader = dataloader_or_dataset
        elif isinstance(dataloader_or_dataset, Dataset):
            loader = DataLoader(
                dataloader_or_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=video_collate_fn,
            )

        if loader is None:
            return {"val_loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0}

        self.model.eval()
        running_loss = 0.0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, dict):
                    x = batch["tensor"].to(self.device, non_blocking=True)
                    y = batch["label"].to(self.device, non_blocking=True)
                elif hasattr(batch, "tensor"):
                    x = batch.tensor.to(self.device)
                    y = torch.tensor([batch.label], device=self.device)
                elif isinstance(batch, (tuple, list)):
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    continue

                with self.amp_handler.autocast():
                    logits = self.model(x)
                    loss = self.loss_fn(logits, y)

                running_loss += loss.item() * x.size(0)
                probs = torch.softmax(logits, dim=-1)[:, 1] if logits.size(-1) > 1 else torch.sigmoid(logits)
                all_probs.append(probs.cpu())
                all_labels.append(y.cpu())

        val_loss = running_loss / len(loader.dataset) if loader.dataset else 0.0

        if all_probs:
            y_true = torch.cat(all_labels, dim=0).numpy()
            y_probs = torch.cat(all_probs, dim=0).numpy()
            eval_metrics = EvaluationMetrics.compute_all(y_true, y_probs)
        else:
            eval_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0}

        eval_metrics["val_loss"] = float(val_loss)
        return eval_metrics

    def save_checkpoint(self, epoch: int, path: Optional[Union[Path, str]] = None) -> Path:
        """Persist trainer and model state checkpoint to disk."""
        filename = Path(path).name if path is not None else f"checkpoint_epoch_{epoch:03d}.pt"
        return self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            loss=self.best_val_loss,
            metrics=self.history,
            scheduler=self.scheduler,
            filename=filename,
        )

    def resume_from_checkpoint(self, checkpoint_path: Union[str, Path]) -> int:
        """Resume training state from saved checkpoint file."""
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        chk = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(chk, dict):
            if "model" in chk:
                self.model.load_state_dict(chk["model"])
            elif "state_dict" in chk:
                self.model.load_state_dict(chk["state_dict"])

            if "optimizer" in chk and self.optimizer is not None:
                self.optimizer.load_state_dict(chk["optimizer"])

            return chk.get("epoch", 0)
        return 0

    def train(self) -> Dict[str, Any]:
        """Execute complete multi-epoch training pipeline."""
        for epoch in range(1, self.config.epochs + 1):
            t_loss = self.train_epoch(epoch)
            self.history["train_loss"].append(t_loss)

            if self.val_loader is not None:
                val_metrics = self.validate()
                v_loss = val_metrics.get("val_loss", 0.0)
                v_acc = val_metrics.get("accuracy", 0.0)
                v_f1 = val_metrics.get("f1", 0.0)

                self.history["val_loss"].append(v_loss)
                self.history["val_accuracy"].append(v_acc)
                self.history["val_f1"].append(v_f1)
                self.history["val_auc"].append(val_metrics.get("auc", 0.0))

                # Save multi-criteria checkpoints
                self.save_checkpoint(epoch, path=self.config.checkpoint_dir / "latest.pt")

                if v_loss < self.best_val_loss:
                    self.best_val_loss = v_loss
                    self.save_checkpoint(epoch, path=self.config.checkpoint_dir / "best_loss.pt")

                if v_acc > self.best_accuracy:
                    self.best_accuracy = v_acc
                    self.save_checkpoint(epoch, path=self.config.checkpoint_dir / "best_accuracy.pt")

                if v_f1 > self.best_f1:
                    self.best_f1 = v_f1
                    self.save_checkpoint(epoch, path=self.config.checkpoint_dir / "best_f1.pt")

                if self.early_stopping(v_loss):
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break

            if self.scheduler is not None:
                if hasattr(self.scheduler, "step"):
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(v_loss)
                    else:
                        self.scheduler.step()

        return self.history


# Class aliases
VideoTrainer = ProductionVideoTrainer
Trainer = ProductionVideoTrainer
