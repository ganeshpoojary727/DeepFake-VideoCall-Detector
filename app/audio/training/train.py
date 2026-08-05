"""Training script orchestrating the AASIST audio deepfake model training pipeline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from app.audio.datasets.dataloader import (
    compute_class_weights,
    create_train_dataloader,
    create_validation_dataloader,
)
from app.audio.models.aasist import AASIST
from app.audio.registry.model_registry import ModelRegistry
from app.audio.training.trainer import Trainer
from app.config.settings import settings
from app.utils.helpers import set_seed
from app.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Entry point for training production AASIST audio model."""
    set_seed(settings.training.seed)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = settings.DEVICE
    logger.info("Using device: %s", device)

    logger.info("─── Hyperparameters (AASIST) ───")
    logger.info("  micro_batch_size  = %d", settings.training.batch_size)
    logger.info("  grad_accum_steps  = %d", settings.training.grad_accum_steps)
    logger.info("  effective_batch   = %d", settings.training.batch_size * settings.training.grad_accum_steps)
    logger.info("  learning_rate     = %f", settings.training.learning_rate)
    logger.info("  epochs            = %d", settings.training.epochs)
    logger.info("  use_amp           = %s", settings.training.use_mixed_precision)
    logger.info("─────────────────────────────────")

    logger.info("Creating data loaders...")
    train_loader = create_train_dataloader()
    validation_loader = create_validation_dataloader()

    logger.info("Instantiating AASIST model...")
    model = ModelRegistry.create(
        "aasist",
        num_classes=settings.model.num_classes,
    ).to(device)

    class_weights = compute_class_weights(train_loader.dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=settings.training.learning_rate,
        weight_decay=settings.training.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=settings.training.scheduler_t0,
    )

    from app.audio.configs.training_config import AudioTrainingConfig
    config = AudioTrainingConfig(
        batch_size=settings.training.batch_size,
        grad_accum_steps=settings.training.grad_accum_steps,
        learning_rate=settings.training.learning_rate,
        epochs=settings.training.epochs,
        use_amp=settings.training.use_mixed_precision,
        gradient_clip_norm=settings.training.gradient_clip_norm,
        weight_decay=settings.training.weight_decay,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=validation_loader,
        config=config,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_path=settings.MODEL_SAVE_PATH,
        use_amp=settings.training.use_mixed_precision,
    )

    trainer.fit(epochs=settings.training.epochs)


if __name__ == "__main__":
    main()