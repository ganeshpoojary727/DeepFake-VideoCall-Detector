"""
Training engine with production best practices.

Features added over v1
──────────────────────
• Early stopping with configurable patience
• Learning rate scheduler (CosineAnnealingWarmRestarts)
• Gradient clipping
• Mixed precision training (torch.cuda.amp)
• TensorBoard logging
• Checkpoint saving and resume
• Best-model saving based on validation **loss** (not accuracy)
• Structured logging via project logger
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────


class EarlyStopping:
    """
    Stop training when validation loss stops improving.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement before stopping.
    min_delta : float
        Minimum change to qualify as an improvement.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.001) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss: float) -> bool:
        """Return ``True`` if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ──────────────────────────────────────────────
# Epoch Metrics
# ──────────────────────────────────────────────


@dataclass
class EpochMetrics:
    """Container for per-epoch training / validation metrics."""

    loss: float = 0.0
    accuracy: float = 0.0
    learning_rate: float = 0.0


# ──────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────


class Trainer:
    """
    Full-featured training engine.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    train_loader : DataLoader
        Training data loader.
    validation_loader : DataLoader
        Validation data loader.
    optimizer : torch.optim.Optimizer
        Optimiser instance.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Compute device.
    scheduler : optional
        Learning-rate scheduler.
    checkpoint_dir : Path | None
        Directory for checkpoint files.
    use_mixed_precision : bool
        Enable AMP (automatic mixed precision).
    gradient_clip_norm : float
        Maximum gradient norm for clipping (0 = disabled).
    early_stopping_patience : int
        Early-stopping patience (0 = disabled).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[object] = None,
        checkpoint_dir: Optional[Path] = None,
        use_mixed_precision: bool = True,
        gradient_clip_norm: float = 1.0,
        early_stopping_patience: int = 5,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.gradient_clip_norm = gradient_clip_norm

        # Mixed precision
        self.use_amp = use_mixed_precision and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Early stopping
        self.early_stopping: Optional[EarlyStopping] = None
        if early_stopping_patience > 0:
            self.early_stopping = EarlyStopping(patience=early_stopping_patience)

        # Checkpointing
        self.checkpoint_dir = checkpoint_dir
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Best model tracking
        self.best_val_loss = float("inf")

        # TensorBoard writer (lazy)
        self._writer = None

    # ── TensorBoard ───────────────────────────

    def _get_writer(self):
        """Lazily initialise a TensorBoard SummaryWriter."""
        if self._writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=str(settings.TENSORBOARD_DIR))
                logger.info("TensorBoard logging to %s", settings.TENSORBOARD_DIR)
            except ImportError:
                logger.info("tensorboard not installed — skipping TB logging")
        return self._writer

    def _log_scalars(self, tag_prefix: str, metrics: EpochMetrics, epoch: int) -> None:
        writer = self._get_writer()
        if writer is None:
            return
        writer.add_scalar(f"{tag_prefix}/loss", metrics.loss, epoch)
        writer.add_scalar(f"{tag_prefix}/accuracy", metrics.accuracy, epoch)
        if metrics.learning_rate > 0:
            writer.add_scalar("lr", metrics.learning_rate, epoch)

    # ── Training ──────────────────────────────

    def train_one_epoch(self) -> EpochMetrics:
        """Run one training epoch and return metrics."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for features, labels in self.train_loader:
            features = features.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # ── Mixed precision forward ───────
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()

                if self.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_norm
                    )

                self.optimizer.step()

            running_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / max(len(self.train_loader), 1)
        epoch_accuracy = (correct / max(total, 1)) * 100

        current_lr = self.optimizer.param_groups[0]["lr"]

        return EpochMetrics(
            loss=epoch_loss, accuracy=epoch_accuracy, learning_rate=current_lr
        )

    # ── Validation ────────────────────────────

    def validate(self) -> EpochMetrics:
        """Run validation and return metrics."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in self.validation_loader:
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(features)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        epoch_loss = running_loss / max(len(self.validation_loader), 1)
        epoch_accuracy = (correct / max(total, 1)) * 100

        return EpochMetrics(loss=epoch_loss, accuracy=epoch_accuracy)

    # ── Full training run ─────────────────────

    def fit(self, epochs: int, save_path: Optional[Path] = None) -> None:
        """
        Execute the full training loop with all bells and whistles.

        Parameters
        ----------
        epochs : int
            Maximum number of epochs.
        save_path : Path | None
            Where to save the best model.  Defaults to ``settings.MODEL_SAVE_PATH``.
        """
        save_path = save_path or settings.MODEL_SAVE_PATH

        logger.info("═" * 50)
        logger.info("Training started — %d epochs, device=%s", epochs, self.device)
        logger.info("Mixed precision: %s", self.use_amp)
        logger.info("Gradient clip norm: %s", self.gradient_clip_norm)
        if self.early_stopping:
            logger.info("Early stopping patience: %d", self.early_stopping.patience)
        logger.info("═" * 50)

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_one_epoch()
            val_metrics = self.validate()

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # TensorBoard
            self._log_scalars("train", train_metrics, epoch)
            self._log_scalars("val", val_metrics, epoch)

            # Best model saving (based on val LOSS, not accuracy)
            if val_metrics.loss < self.best_val_loss:
                self.best_val_loss = val_metrics.loss
                torch.save(self.model.state_dict(), save_path)
                logger.info("✅ Best model saved (val_loss=%.4f)", val_metrics.loss)

            # Logging
            logger.info(
                "Epoch [%d/%d] | "
                "Train Loss: %.4f  Acc: %.2f%% | "
                "Val Loss: %.4f  Acc: %.2f%% | "
                "LR: %.6f",
                epoch,
                epochs,
                train_metrics.loss,
                train_metrics.accuracy,
                val_metrics.loss,
                val_metrics.accuracy,
                train_metrics.learning_rate,
            )

            # Early stopping
            if self.early_stopping and self.early_stopping(val_metrics.loss):
                logger.info(
                    "⏹ Early stopping triggered at epoch %d (patience=%d)",
                    epoch,
                    self.early_stopping.patience,
                )
                break

        # Cleanup TensorBoard
        writer = self._get_writer()
        if writer is not None:
            writer.close()

        logger.info("═" * 50)
        logger.info("Training complete — best val_loss: %.4f", self.best_val_loss)
        logger.info("Model saved to: %s", save_path)
        logger.info("═" * 50)

    # ── Checkpoint save / load ────────────────

    def save_checkpoint(self, epoch: int, path: Path) -> None:
        """Save a full training checkpoint (model + optimizer + epoch)."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(checkpoint, path)
        logger.info("Checkpoint saved: %s", path)

    def load_checkpoint(self, path: Path) -> int:
        """
        Load a training checkpoint and return the last completed epoch.

        Returns
        -------
        int
            The epoch number to resume from.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint["epoch"]
        logger.info("Checkpoint loaded from epoch %d", epoch)
        return epoch