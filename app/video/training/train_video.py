"""
Training script for Video DeepFake Detection model (VideoDeepFakeCNN).

Usage::

    python -m app.video.training.train_video --data-dir path/to/video_dataset

Expects data directory structure:
    data_dir/
      real/   (real face images or videos)
      fake/   (deepfake face images or videos)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from app.video.datasets.video_dataset import VideoDeepFakeDataset
from app.video.models.video_model import VideoDeepFakeCNN
from app.config.settings import settings
from app.utils.helpers import set_seed
from app.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    default_dir = Path("datasets/video/FaceForensics++_C23")
    if not default_dir.exists():
        default_dir = Path("datasets/video")

    parser = argparse.ArgumentParser(description="Train Video DeepFake Detection Model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(default_dir),
        help="Path to directory containing real/fake or FaceForensics++ subdirectories",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap max samples per category")
    args, _ = parser.parse_known_args()

    set_seed(settings.training.seed)
    device = settings.DEVICE
    logger.info("Training VideoDeepFakeCNN on device: %s", device)

    dataset_path = Path(args.data_dir)
    if not dataset_path.exists():
        logger.error(
            "Video dataset directory not found: %s\n"
            "Organise your data as:\n"
            "  %s/real/\n"
            "  %s/fake/\n",
            dataset_path, dataset_path, dataset_path
        )
        print(f"❌ Dataset directory not found: {dataset_path}")
        print("To train the video model, create:")
        print(f"  {dataset_path}/real/")
        print(f"  {dataset_path}/fake/")
        return

    full_dataset = VideoDeepFakeDataset(
        dataset_path,
        transform=True,
        max_samples_per_category=args.max_samples,
    )
    if len(full_dataset) == 0:
        logger.error("No valid samples found in %s", dataset_path)
        return

    # 80/20 train/val split
    val_size = max(1, int(len(full_dataset) * 0.2))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = VideoDeepFakeCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    save_path = settings.MODEL_DIR / "video_model.pth"

    logger.info("Starting training (%d epochs, %d train, %d val)...", args.epochs, train_size, val_size)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_correct = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            train_correct += (out.argmax(1) == y).sum().item()

        # Validation
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * len(y)
                val_correct += (out.argmax(1) == y).sum().item()

        val_acc = val_correct / val_size
        logger.info(
            "Epoch %2d/%2d — Train Loss: %.4f, Val Loss: %.4f, Val Acc: %.2f%%",
            epoch, args.epochs, train_loss / train_size, val_loss / val_size, val_acc * 100
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            logger.info("Saved best video model to %s (acc: %.2f%%)", save_path, val_acc * 100)

    print(f"\n[OK] Video model training complete! Saved to {save_path} (Best Acc: {best_acc:.1%})")


if __name__ == "__main__":
    main()
