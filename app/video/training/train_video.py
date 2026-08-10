"""Training script for Video DeepFake Detection model (EfficientNet-B4 + Temporal Attention)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
import torch

from app.config.settings import settings
from app.utils.helpers import set_seed
from app.utils.logger import get_logger
from app.video.builders import VideoTrainerBuilder
from app.video.configs import DatasetConfig, ModelConfig, VideoTrainingConfig
from app.video.datasets import VideoDataset
from app.video.datasets.dataloader import create_train_dataloader, create_validation_dataloader

logger = get_logger(__name__)


def resolve_ffpp_directories(data_dir: str | Path) -> Tuple[Path, Path]:
    """Resolve original (real) and fake video subdirectories for FaceForensics++ dataset."""
    p = Path(data_dir)
    search_paths = [
        p,
        p / "faceforensics",
        Path("datasets/video/faceforensics"),
        Path("datasets/video"),
    ]
    for base in search_paths:
        if (base / "original").exists():
            real_dir = base / "original"
            if (base / "Deepfakes").exists():
                fake_dir = base / "Deepfakes"
            else:
                fakes = [
                    d for d in base.glob("*")
                    if d.is_dir() and d.name.lower() not in ("original", "csv", "cache", "processed")
                ]
                fake_dir = fakes[0] if fakes else None
            if fake_dir and fake_dir.exists():
                return real_dir, fake_dir

    raise FileNotFoundError(
        f"Could not locate valid FaceForensics++ subdirectories ('original' and fake folders) "
        f"under '{data_dir}' or fallback paths."
    )


def build_samples_split(real_dir: Path, fake_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build deterministic 80/20 train/validation sample splits based on video subject ID."""
    real_videos = sorted(list(real_dir.glob("*.mp4")))
    fake_videos = sorted(list(fake_dir.glob("*.mp4")))

    train_samples: List[Dict[str, Any]] = []
    val_samples: List[Dict[str, Any]] = []

    for p in real_videos:
        try:
            vid_id = int(p.stem)
        except ValueError:
            vid_id = abs(hash(p.stem)) % 100
        sample = {"filepath": str(p), "label": 0, "sample_id": p.name}
        if (vid_id % 100) < 80:
            train_samples.append(sample)
        else:
            val_samples.append(sample)

    for p in fake_videos:
        parts = p.stem.split("_")
        try:
            primary_id = int(parts[0])
        except ValueError:
            primary_id = abs(hash(p.stem)) % 100
        sample = {"filepath": str(p), "label": 1, "sample_id": p.name}
        if (primary_id % 100) < 80:
            train_samples.append(sample)
        else:
            val_samples.append(sample)

    return train_samples, val_samples


def main() -> Dict[str, Any]:
    """Entry point for training production EfficientNet-B4 + Temporal Attention video model."""
    default_dir = Path("datasets/video/FaceForensics++_C23")
    if not default_dir.exists():
        default_dir = Path("datasets/video/faceforensics")

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
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples per split for quick smoke tests (0=all)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader worker processes")
    args, _ = parser.parse_known_args()

    set_seed(settings.training.seed)
    device = settings.DEVICE
    logger.info("Training EfficientNetB4Model on device: %s", device)

    # 1. Resolve dataset directories & build split
    real_dir, fake_dir = resolve_ffpp_directories(args.data_dir)
    train_samples, val_samples = build_samples_split(real_dir, fake_dir)

    if args.max_samples > 0:
        train_reals = [s for s in train_samples if s["label"] == 0][: args.max_samples // 2]
        train_fakes = [s for s in train_samples if s["label"] == 1][: args.max_samples // 2]
        train_samples = train_reals + train_fakes

        val_reals = [s for s in val_samples if s["label"] == 0][: max(1, args.max_samples // 8)]
        val_fakes = [s for s in val_samples if s["label"] == 1][: max(1, args.max_samples // 8)]
        val_samples = val_reals + val_fakes

    train_real_cnt = sum(1 for s in train_samples if s["label"] == 0)
    train_fake_cnt = sum(1 for s in train_samples if s["label"] == 1)
    val_real_cnt = sum(1 for s in val_samples if s["label"] == 0)
    val_fake_cnt = sum(1 for s in val_samples if s["label"] == 1)

    logger.info(
        f"Dataset Split -> Train: {len(train_samples)} (Real: {train_real_cnt}, Fake: {train_fake_cnt}) | "
        f"Val: {len(val_samples)} (Real: {val_real_cnt}, Fake: {val_fake_cnt})"
    )

    # 2. Build datasets and dataloaders
    dataset_cfg = DatasetConfig(
        dataset_name="faceforensics",
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

    # 3. Configure model and trainer
    cfg_train = VideoTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=str(device),
        use_amp=torch.cuda.is_available(),
    )
    cfg_model = ModelConfig(
        backbone_name="efficientnet_b4",
        attention_name="temporal_transformer",
        num_classes=2,
        freeze_backbone=True,
        pretrained=True,
        sequence_length=16,
        dropout=0.2,
    )

    trainer = (
        VideoTrainerBuilder()
        .with_training_config(cfg_train)
        .with_model_config(cfg_model)
        .with_dataset_config(dataset_cfg)
        .with_dataloaders(train_loader, val_loader)
        .build()
    )

    logger.info("Starting Video Model Training Pipeline...")
    history = trainer.train()

    # Save training history JSON to checkpoint directory
    import json
    history_file = cfg_train.checkpoint_dir / "training_history.json"
    try:
        cfg_train.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Saved training history to {history_file}")
    except Exception as err:
        logger.warning(f"Could not save training history: {err}")

    logger.info("Video Model Training Pipeline Completed.")
    return history


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()

