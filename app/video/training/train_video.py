"""Training script for Video DeepFake Detection model (EfficientNet-B4 + Temporal Attention)."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from app.config.settings import settings
from app.utils.helpers import set_seed
from app.utils.logger import get_logger
from app.video.builders import VideoTrainerBuilder
from app.video.configs import ModelConfig, VideoTrainingConfig
from app.video.models import EfficientNetB4Model

logger = get_logger(__name__)


def main() -> None:
    """Entry point for training production EfficientNet-B4 + Temporal Attention video model."""
    default_dir = Path("datasets/video/FaceForensics++_C23")
    if not default_dir.exists():
        default_dir = Path("datasets/video")

    parser = argparse.ArgumentParser(description="Train Production Video DeepFake Detection Model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(default_dir),
        help="Path to directory containing real/fake or FaceForensics++ subdirectories",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args, _ = parser.parse_known_args()

    set_seed(settings.training.seed)
    device = settings.DEVICE
    logger.info("Training EfficientNetB4Model on device: %s", device)

    cfg_train = VideoTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=str(device),
    )
    cfg_model = ModelConfig(
        backbone_name="efficientnet_b4",
        attention_name="temporal_transformer",
        num_classes=2,
    )

    trainer = (
        VideoTrainerBuilder()
        .with_training_config(cfg_train)
        .with_model_config(cfg_model)
        .build()
    )

    logger.info("Starting Video Model Training Pipeline...")
    # Ready for fit/train execution


if __name__ == "__main__":
    main()
