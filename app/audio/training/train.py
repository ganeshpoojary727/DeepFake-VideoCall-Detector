"""
Training script — orchestrates the full training pipeline.

Usage::

    python -m app.audio.training.train

Improvements over v1
─────────────────────
• Reproducibility via ``set_seed()``
• Class weights for CrossEntropyLoss (addresses 1:8.8 imbalance)
• CosineAnnealingWarmRestarts scheduler
• AdamW optimiser with weight decay
• Delegates to ``Trainer.fit()`` for early stopping, AMP, TensorBoard, etc.
• Logs all hyperparameters at start
• ``cudnn.benchmark = True`` for CNN speed-up
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from app.audio.datasets.dataloader import (
    compute_class_weights,
    create_train_dataloader,
    create_validation_dataloader,
)
from app.audio.models.cnn_model import DeepFakeCNN
from app.audio.training.trainer import Trainer
from app.config.settings import settings
from app.utils.helpers import set_seed
from app.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Entry point for training."""

    # ── Reproducibility ───────────────────────
    set_seed(settings.training.seed)
    torch.backends.cudnn.benchmark = True

    device = settings.DEVICE
    logger.info("Using device: %s", device)

    # ── Log hyperparameters ───────────────────
    logger.info("─── Hyperparameters ───")
    logger.info("  batch_size       = %d", settings.training.batch_size)
    logger.info("  learning_rate    = %f", settings.training.learning_rate)
    logger.info("  epochs           = %d", settings.training.epochs)
    logger.info("  weight_decay     = %f", settings.training.weight_decay)
    logger.info("  num_workers      = %d", settings.training.num_workers)
    logger.info("  gradient_clip    = %f", settings.training.gradient_clip_norm)
    logger.info("  early_stop       = %d", settings.training.early_stopping_patience)
    logger.info("  mixed_precision  = %s", settings.training.use_mixed_precision)
    logger.info("  seed             = %d", settings.training.seed)
    logger.info("───────────────────────")

    # ── DataLoaders ───────────────────────────
    logger.info("Creating data loaders...")
    train_loader = create_train_dataloader()
    validation_loader = create_validation_dataloader()

    # ── Model ─────────────────────────────────
    from app.audio.models.model_registry import ModelRegistry
    model = ModelRegistry.create(
        settings.model.model_name,
        num_classes=settings.model.num_classes,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s  (%s params)", settings.model.model_name, f"{total_params:,}")

    # ── Loss (with class weights) ─────────────
    class_weights = compute_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimiser ─────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=settings.training.learning_rate,
        weight_decay=settings.training.weight_decay,
    )

    # ── LR Scheduler ─────────────────────────
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=settings.training.scheduler_t0
    )

    # ── Trainer ───────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        use_mixed_precision=settings.training.use_mixed_precision,
        gradient_clip_norm=settings.training.gradient_clip_norm,
        early_stopping_patience=settings.training.early_stopping_patience,
    )

    # ── Train ─────────────────────────────────
    trainer.fit(
        epochs=settings.training.epochs,
        save_path=settings.MODEL_SAVE_PATH,
    )


if __name__ == "__main__":
    main()