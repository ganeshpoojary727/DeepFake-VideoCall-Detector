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

    device = settings.DEVICE
    logger.info("Using device: %s", device)

    logger.info("─── Hyperparameters (AASIST) ───")
    logger.info("  batch_size       = %d", settings.training.batch_size)
    logger.info("  learning_rate    = %f", settings.training.learning_rate)
    logger.info("  epochs           = %d", settings.training.epochs)
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

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
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