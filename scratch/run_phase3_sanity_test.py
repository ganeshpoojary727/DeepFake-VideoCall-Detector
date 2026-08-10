"""Phase 3 - Video Model Training Sanity Test Script."""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.video.configs.dataset_config import DatasetConfig
from app.video.configs.model_config import ModelConfig
from app.video.configs.training_config import VideoTrainingConfig
from app.video.datasets.dataloader import video_collate_fn
from app.video.datasets.video_dataset import VideoDataset
from app.video.evaluation.metrics import EvaluationMetrics
from app.video.models.efficientnet.model import EfficientNetB4Model
from app.video.training.optimizer_factory import OptimizerFactory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants
SEED = 42
SANITY_TEST_DIR = PROJECT_ROOT / "trained_models" / "video" / "sanity_test"
AUDIO_CKPT_PATH = PROJECT_ROOT / "trained_models" / "audio" / "best_model.pt"
FFPP_ORIGINAL_DIR = PROJECT_ROOT / "datasets" / "video" / "faceforensics" / "original"
FFPP_DEEPFAKES_DIR = PROJECT_ROOT / "datasets" / "video" / "faceforensics" / "Deepfakes"


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def assert_finite(tensor: torch.Tensor, name: str) -> None:
    """Ensure tensor contains no NaN or Inf values."""
    assert torch.isfinite(tensor).all(), (
        f"[{name}] Non-finite values detected! NaN: {torch.isnan(tensor).sum().item()}, "
        f"Inf: {torch.isinf(tensor).sum().item()}"
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> dict:
    """Run evaluation pass and compute metrics."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            x = batch["tensor"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            assert_finite(logits, "eval_logits")
            assert_finite(loss, "eval_loss")

            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    y_true = torch.cat(all_labels, dim=0).numpy()
    y_probs = torch.cat(all_probs, dim=0).numpy()

    metrics = EvaluationMetrics.compute_all(y_true, y_probs, threshold=0.5)
    metrics["loss"] = float(avg_loss)
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool = True,
) -> dict:
    """Run single epoch of training and compute train metrics."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_probs = []
    all_labels = []

    for batch in loader:
        x = batch["tensor"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        assert_finite(logits, "train_logits")
        assert_finite(loss, "train_loss")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_probs.append(probs.detach().cpu())
            all_labels.append(y.detach().cpu())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    y_true = torch.cat(all_labels, dim=0).numpy()
    y_probs = torch.cat(all_probs, dim=0).numpy()

    metrics = EvaluationMetrics.compute_all(y_true, y_probs, threshold=0.5)
    metrics["loss"] = float(avg_loss)
    return metrics


def run_sanity_test():
    print("==================================================")
    print("  PHASE 3 — VIDEO MODEL TRAINING SANITY TEST")
    print("==================================================")

    set_seed(SEED)

    # 1. DATA PREPARATION
    real_videos = sorted(list(FFPP_ORIGINAL_DIR.glob("*.mp4")))
    fake_videos = sorted(list(FFPP_DEEPFAKES_DIR.glob("*.mp4")))

    print(f"Total available real videos: {len(real_videos)}")
    print(f"Total available fake videos: {len(fake_videos)}")

    # Target 100 real videos (000.mp4 .. 099.mp4) & 100 fake videos from Deepfakes
    selected_real = real_videos[:100]
    selected_fake = fake_videos[:100]

    # Split 80% train / 20% val by video ID prefix to ensure disjoint subjects
    # IDs 000..079 -> Train (80 real, 80 fake)
    # IDs 080..099 -> Val (20 real, 20 fake)
    train_samples = []
    val_samples = []

    for p in selected_real:
        vid_id = int(p.stem)
        sample = {"filepath": str(p), "label": 0, "sample_id": p.name}
        if vid_id < 80:
            train_samples.append(sample)
        else:
            val_samples.append(sample)

    for p in selected_fake:
        parts = p.stem.split("_")
        primary_id = int(parts[0])
        sample = {"filepath": str(p), "label": 1, "sample_id": p.name}
        if primary_id < 80:
            train_samples.append(sample)
        else:
            val_samples.append(sample)

    train_real_cnt = sum(1 for s in train_samples if s["label"] == 0)
    train_fake_cnt = sum(1 for s in train_samples if s["label"] == 1)
    val_real_cnt = sum(1 for s in val_samples if s["label"] == 0)
    val_fake_cnt = sum(1 for s in val_samples if s["label"] == 1)

    print(f"\nDataset Split (Deterministic 80/20 by Video ID):")
    print(f"Train Videos: {len(train_samples)} (Real: {train_real_cnt}, Fake: {train_fake_cnt})")
    print(f"Val Videos:   {len(val_samples)} (Real: {val_real_cnt}, Fake: {val_fake_cnt})")

    # Audio checkpoint safety check
    assert AUDIO_CKPT_PATH.exists(), f"Audio checkpoint missing at {AUDIO_CKPT_PATH}"
    audio_mtime_before = AUDIO_CKPT_PATH.stat().st_mtime
    audio_size_before = AUDIO_CKPT_PATH.stat().st_size

    # Datasets and Loaders
    dataset_cfg = DatasetConfig(
        sequence_length=16,
        target_resolution=(224, 224),
        crop_faces=True,
        sampling_strategy="uniform",
    )

    train_dataset = VideoDataset(config=dataset_cfg, samples=train_samples)
    val_dataset = VideoDataset(config=dataset_cfg, samples=val_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=video_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=video_collate_fn,
    )

    # Model Initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    model_cfg = ModelConfig(
        pretrained=True,
        freeze_backbone=True,
        num_classes=2,
    )
    model = EfficientNetB4Model(config=model_cfg).to(device)

    # Verify backbone frozen status
    trainable_params = model.get_trainable_parameters()
    total_params = model.get_num_parameters()
    frozen_params = total_params - trainable_params

    print(f"\nModel Configuration:")
    print(f"Backbone: EfficientNet-B4 (Frozen: {model_cfg.freeze_backbone})")
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Frozen Parameters:    {frozen_params:,}")

    for p in model.backbone.parameters():
        assert not p.requires_grad, "Backbone parameter requires_grad is True while frozen!"

    for p in model.temporal_encoder.parameters():
        assert p.requires_grad, "Temporal encoder parameter requires_grad is False!"

    for p in model.classifier.parameters():
        assert p.requires_grad, "Classifier parameter requires_grad is False!"

    # Training Setup
    train_cfg = VideoTrainingConfig(
        epochs=5,
        batch_size=4,
        learning_rate=1e-4,
        weight_decay=1e-4,
        optimizer_name="adamw",
        use_amp=True,
    )

    optimizer = OptimizerFactory.create_optimizer(model, config=train_cfg)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg.use_amp)

    os.makedirs(SANITY_TEST_DIR, exist_ok=True)

    history = {
        "config": {
            "num_frames": 16,
            "resolution": "224x224",
            "batch_size": 4,
            "optimizer": "AdamW",
            "learning_rate": train_cfg.learning_rate,
            "weight_decay": train_cfg.weight_decay,
            "loss": "CrossEntropyLoss",
            "epochs": 5,
            "freeze_backbone": True,
            "seed": SEED,
            "device": str(device),
            "use_amp": train_cfg.use_amp,
        },
        "dataset_split": {
            "total_train": len(train_samples),
            "train_real": train_real_cnt,
            "train_fake": train_fake_cnt,
            "total_val": len(val_samples),
            "val_real": val_real_cnt,
            "val_fake": val_fake_cnt,
            "train_video_paths": [s["filepath"] for s in train_samples],
            "val_video_paths": [s["filepath"] for s in val_samples],
        },
        "epoch_metrics": [],
    }

    print("\n--- INITIAL EVALUATION (EPOCH 0) ---")
    t0_eval = time.perf_counter()
    epoch0_val = evaluate(model, val_loader, criterion, device, use_amp=train_cfg.use_amp)
    t1_eval = time.perf_counter()

    vram_mb_ep0 = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0

    print(
        f"Epoch 0 (Val Only): Loss={epoch0_val['loss']:.4f}, Acc={epoch0_val['accuracy']:.4f}, "
        f"Prec={epoch0_val['precision']:.4f}, Rec={epoch0_val['recall']:.4f}, F1={epoch0_val['f1']:.4f}, "
        f"AUC={epoch0_val['auc']:.4f} [Time: {t1_eval - t0_eval:.2f}s, VRAM: {vram_mb_ep0:.2f} MB]"
    )

    history["epoch_metrics"].append({
        "epoch": 0,
        "duration_seconds": round(t1_eval - t0_eval, 2),
        "learning_rate": train_cfg.learning_rate,
        "vram_mb": round(vram_mb_ep0, 2),
        "train": None,
        "val": epoch0_val,
    })

    best_val_loss = float("inf")
    best_epoch = -1

    # 5-Epoch Training Loop
    print("\n--- STARTING 5-EPOCH SANITY TRAINING ---")
    for epoch in range(1, train_cfg.epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        gc.collect()

        t0_ep = time.perf_counter()
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp=train_cfg.use_amp)
        val_m = evaluate(model, val_loader, criterion, device, use_amp=train_cfg.use_amp)
        t1_ep = time.perf_counter()

        duration = t1_ep - t0_ep
        vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/5: "
            f"Train [Loss: {train_m['loss']:.4f}, Acc: {train_m['accuracy']:.4f}, Prec: {train_m['precision']:.4f}, Rec: {train_m['recall']:.4f}, F1: {train_m['f1']:.4f}] | "
            f"Val [Loss: {val_m['loss']:.4f}, Acc: {val_m['accuracy']:.4f}, Prec: {val_m['precision']:.4f}, Rec: {val_m['recall']:.4f}, F1: {val_m['f1']:.4f}, AUC: {val_m['auc']:.4f}] | "
            f"Duration: {duration:.2f}s | VRAM: {vram_mb:.2f} MB",
            flush=True,
        )

        ep_record = {
            "epoch": epoch,
            "duration_seconds": round(duration, 2),
            "learning_rate": current_lr,
            "vram_mb": round(vram_mb, 2),
            "train": train_m,
            "val": val_m,
        }
        history["epoch_metrics"].append(ep_record)

        # Save best model checkpoint
        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_epoch = epoch
            best_ckpt_path = SANITY_TEST_DIR / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_m["loss"],
                    "val_accuracy": val_m["accuracy"],
                    "val_f1": val_m["f1"],
                    "config": history["config"],
                },
                best_ckpt_path,
            )
            print(f"  --> Saved new best checkpoint at Epoch {epoch} to {best_ckpt_path}")

    # Save final model checkpoint
    final_ckpt_path = SANITY_TEST_DIR / "final_model.pt"
    torch.save(
        {
            "epoch": 5,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": history["epoch_metrics"][-1]["val"]["loss"],
            "val_accuracy": history["epoch_metrics"][-1]["val"]["accuracy"],
            "val_f1": history["epoch_metrics"][-1]["val"]["f1"],
            "config": history["config"],
        },
        final_ckpt_path,
    )
    print(f"Saved final checkpoint to {final_ckpt_path}")

    # Save training history JSON
    history_path = SANITY_TEST_DIR / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_path}")

    # Post-training safety verification: check audio checkpoint
    audio_mtime_after = AUDIO_CKPT_PATH.stat().st_mtime
    audio_size_after = AUDIO_CKPT_PATH.stat().st_size
    assert audio_mtime_before == audio_mtime_after and audio_size_before == audio_size_after, (
        f"CRITICAL SAFETY VIOLATION: Audio checkpoint {AUDIO_CKPT_PATH} was modified!"
    )
    print(f"\nAudio Checkpoint Integrity Check: UNTOUCHED & VERIFIED [PASSED]")

    print("\n==================================================")
    print("  SANITY TEST EXECUTION COMPLETE")
    print("==================================================")

    return history


if __name__ == "__main__":
    run_sanity_test()
