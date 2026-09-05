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


BELOW_NORMAL_PRIORITY_CLASS = 0x00004000


def set_smooth_process_priority() -> None:
    """Set Windows process priority to BELOW_NORMAL so the desktop and UI remain smooth and lag-free."""
    if sys.platform == "win32":
        try:
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS)
            logger.info("Windows process priority set to BELOW_NORMAL (desktop will stay smooth & responsive)")
        except Exception as err:
            logger.warning("Could not set process priority: %s", err)


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
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (utilizes ~4-5GB VRAM on RTX 4050)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (default 0 for crash-free Windows execution with pre-cached frames)")
    parser.add_argument("--target-per-class", type=int, default=1800, help="Target samples per class")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--full-val", action="store_true", default=False, help="Run full validation set every epoch instead of fast subset")
    parser.add_argument("--val-max-batches", type=int, default=25, help="Max validation batches for intermediate epochs (default: 25 = 200 videos ~35s)")
    args, _ = parser.parse_known_args()

    # Limit CPU thread hogging so UI and OS desktop stay completely smooth
    import os
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENCV_LOG_LEVEL"] = "OFF"
    torch.set_num_threads(2)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 0. Configure native log file
    log_dir = settings.project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"overnight_training_{time.strftime('%Y%m%d_%H%M%S')}.log"
    import logging
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)

    # 1. Keep PC awake and make it smooth
    prevent_windows_sleep()
    set_smooth_process_priority()

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

    val_batches = None if args.full_val else args.val_max_batches
    cfg_train = VideoTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=str(device),
        use_amp=torch.cuda.is_available(),
        checkpoint_dir=checkpoint_dir,
        val_max_batches=val_batches,
    )

    cfg_model = ModelConfig(
        backbone_name="efficientnet_b4",
        attention_name="temporal_transformer",
        num_classes=2,
        freeze_backbone=False,
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

    # Partial fine-tuning: freeze low-level stages 0..4, train high-level stages 5..7 + temporal transformer
    # This utilizes ~4.0 GB VRAM on your RTX 4050 and drastically accelerates GPU tensor parallelism
    trainer.model.backbone.freeze_layers(until_stage=5)
    import torch.optim as optim
    trainable_params = [p for p in trainer.model.parameters() if p.requires_grad]
    trainer.optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    trainer.scheduler = optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=args.epochs)
    logger.info("⚡ GPU Acceleration Configured: %s trainable parameters (~4.0 GB VRAM target)", f"{sum(p.numel() for p in trainable_params):,}")

    # 6. Execute Training (with automatic or explicit resume)
    resume_target = args.resume
    if resume_target is None and (checkpoint_dir / "latest.pt").exists():
        resume_target = str(checkpoint_dir / "latest.pt")
        logger.info("Found existing checkpoint '%s' — auto-resuming training seamlessly.", resume_target)

    if resume_target:
        resume_path = Path(resume_target)
        if resume_path.exists():
            resumed_epoch = trainer.resume_from_checkpoint(resume_path)
            logger.info("  • Resumed training from %s at epoch %d", resume_path, resumed_epoch)
        else:
            logger.warning("  • Checkpoint for resume not found at %s. Starting fresh.", resume_path)

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
