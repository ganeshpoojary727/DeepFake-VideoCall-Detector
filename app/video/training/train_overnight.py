"""
Overnight Video Deepfake Model Training Orchestrator.

Key Features for Unattended Overnight Execution:
1. Prevents Windows laptop/PC sleep/suspend during training via Win32 API.
2. Trains on balanced multi-benchmark data: FaceForensics++ (all 6 methods) + Celeb-DF v2.
3. Automatically checkpoints best models (best_loss, best_accuracy, best_auc, latest).
4. Logs full progress to console and logs/overnight_training.log.
5. Safely restores system power state upon completion or unexpected interruption.
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch

from app.config.settings import settings
from app.utils.helpers import set_seed
from app.utils.logger import get_logger
from app.video.builders import VideoTrainerBuilder
from app.video.configs import DatasetConfig, ModelConfig, VideoTrainingConfig
from app.video.datasets import VideoDataset
from app.video.datasets.dataloader import create_train_dataloader, create_validation_dataloader
from app.video.training.multi_dataset_loader import build_balanced_multi_dataset

logger = get_logger(__name__)

# Windows API constants to keep PC awake
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040


def prevent_windows_sleep() -> None:
    """Instruct Windows power manager to not enter sleep while training."""
    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
            )
            logger.info("Windows sleep prevention ENABLED (PC will remain awake during training)")
        except Exception as err:
            logger.warning("Could not set Windows execution state: %s", err)


def allow_windows_sleep() -> None:
    """Restore default Windows sleep state when training finishes."""
    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            logger.info("Windows sleep prevention DISABLED (normal power management restored)")
        except Exception as err:
            logger.warning("Could not restore Windows execution state: %s", err)


atexit.register(allow_windows_sleep)


def main() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Overnight Video DeepFake Training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (fits 6GB VRAM on RTX 4050)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--target-per-class", type=int, default=1800, help="Target samples per class")
    args, _ = parser.parse_known_args()

    # 1. Keep PC awake
    prevent_windows_sleep()

    set_seed(42)
    device = settings.DEVICE
    logger.info("=" * 65)
    logger.info("  🚀 STARTING OVERNIGHT DEEPFAKE MODEL TRAINING PIPELINE")
    logger.info("  • Device: %s (CUDA: %s)", device, torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("  • GPU: %s", torch.cuda.get_device_name(0))
        logger.info("  • VRAM: %.2f GB", torch.cuda.get_device_properties(0).total_memory / (1024**3))
    logger.info("  • Epochs: %d | Batch Size: %d | Learning Rate: %.1e", args.epochs, args.batch_size, args.lr)
    logger.info("=" * 65)

    # 2. Build multi-dataset split combining FF++ and Celeb-DF
    datasets_root = settings.project_root / "datasets"
    train_samples, val_samples = build_balanced_multi_dataset(
        datasets_root=datasets_root,
        target_samples_per_class=args.target_per_class,
        train_ratio=0.80,
        seed=42,
    )

    # 3. Create Dataset and DataLoader
    dataset_cfg = DatasetConfig(
        dataset_name="multi_benchmark",
        sequence_length=16,
        target_resolution=(224, 224),
        crop_faces=True,
        sampling_strategy="uniform",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    train_dataset = VideoDataset(config=dataset_cfg, samples=train_samples)
    val_dataset = VideoDataset(config=dataset_cfg, samples=val_samples)

    train_loader = create_train_dataloader(train_dataset, config=dataset_cfg)
    val_loader = create_validation_dataloader(val_dataset, config=dataset_cfg)

    # 4. Training configuration
    checkpoint_dir = settings.project_root / "trained_models" / "video"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cfg_train = VideoTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=str(device),
        use_amp=torch.cuda.is_available(),
        checkpoint_dir=checkpoint_dir,
    )

    cfg_model = ModelConfig(
        backbone_name="efficientnet_b4",
        attention_name="temporal_transformer",
        num_classes=2,
        freeze_backbone=True,
        pretrained=True,
        sequence_length=16,
        dropout=0.25,
    )

    # 5. Build Trainer
    trainer = (
        VideoTrainerBuilder()
        .with_training_config(cfg_train)
        .with_model_config(cfg_model)
        .with_dataset_config(dataset_cfg)
        .with_dataloaders(train_loader, val_loader)
        .build()
    )

    # 6. Execute Training
    start_time = time.time()
    try:
        history = trainer.train()
        total_hours = (time.time() - start_time) / 3600.0

        # Save history to JSON
        history_file = checkpoint_dir / "training_history.json"
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        logger.info("=" * 65)
        logger.info("  ✅ OVERNIGHT TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("  • Total Runtime: %.2f hours", total_hours)
        logger.info("  • Best Checkpoints Saved in: %s", checkpoint_dir)
        logger.info("  • Training History JSON: %s", history_file)
        logger.info("=" * 65)
        return history

    except KeyboardInterrupt:
        logger.warning("Training manually interrupted by user.")
        return {}
    except Exception as exc:
        logger.exception("Unexpected error occurred during overnight training: %s", exc)
        raise
    finally:
        allow_windows_sleep()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
