"""
Read-Only Evaluation Script for Video Checkpoints.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, confusion_matrix

from app.config.settings import settings
from app.utils.helpers import set_seed
from app.video.configs import DatasetConfig, ModelConfig
from app.video.datasets import VideoDataset
from app.video.datasets.dataloader import create_validation_dataloader
from app.video.models import EfficientNetB4Model
from app.video.training.train_video import resolve_ffpp_directories, build_samples_split


def compute_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    return eer, thresholds[eer_idx]


def evaluate_all():
    set_seed(settings.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # 1. Dataset & DataLoader setup
    real_dir, fake_dir = resolve_ffpp_directories("datasets/video/FaceForensics++_C23")
    _, val_samples = build_samples_split(real_dir, fake_dir)

    dataset_cfg = DatasetConfig(
        dataset_name="faceforensics",
        sequence_length=16,
        target_resolution=(224, 224),
        crop_faces=True,
        sampling_strategy="uniform",
        batch_size=8,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    val_dataset = VideoDataset(config=dataset_cfg, samples=val_samples)
    val_loader = create_validation_dataloader(val_dataset, config=dataset_cfg)

    n_reals = sum(1 for s in val_samples if s["label"] == 0)
    n_fakes = sum(1 for s in val_samples if s["label"] == 1)
    print(f"Validation set size: {len(val_samples)} (Reals: {n_reals}, Fakes: {n_fakes})", flush=True)

    ckpt_files = sorted(Path("trained_models/video").glob("*.pt"))

    cfg_model = ModelConfig(
        backbone_name="efficientnet_b4",
        attention_name="temporal_transformer",
        num_classes=2,
        freeze_backbone=True,
        pretrained=False,
        sequence_length=16,
        dropout=0.2,
    )

    for ckpt_path in ckpt_files:
        print("\n" + "=" * 60, flush=True)
        print(f"EVALUATING CHECKPOINT: {ckpt_path.name}", flush=True)
        print("=" * 60, flush=True)

        hash_before = compute_md5(ckpt_path)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        epoch = ckpt.get("epoch", "unknown")
        state_dict = ckpt.get("model_state", ckpt)

        nan_params = sum(torch.isnan(p).sum().item() for p in state_dict.values() if isinstance(p, torch.Tensor))
        inf_params = sum(torch.isinf(p).sum().item() for p in state_dict.values() if isinstance(p, torch.Tensor))
        print(f"Epoch in checkpoint: {epoch}", flush=True)
        print(f"Checkpoint param NaN count: {nan_params}, Inf count: {inf_params}", flush=True)

        model = EfficientNetB4Model(cfg_model)
        load_res = model.load_state_dict(state_dict, strict=True)
        print(f"Strict load result - Missing keys: {len(load_res.missing_keys)}, Unexpected keys: {len(load_res.unexpected_keys)}", flush=True)

        model.to(device)
        model.eval()

        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(val_loader):
                    print(f"  Processed batch {batch_idx+1}/{len(val_loader)}", flush=True)
                frames = batch["tensor"].to(device)
                labels = batch["label"].to(device)
                outputs = model(frames)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_targets.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        y_true = np.array(all_targets)
        y_scores = np.array(all_probs)

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        j_scores = tpr - fpr
        opt_idx = np.argmax(j_scores)
        opt_thresh = thresholds[opt_idx]

        eer, eer_thresh = compute_eer(y_true, y_scores)

        print(f"ROC-AUC: {roc_auc:.4f}", flush=True)
        print(f"EER: {eer:.4f} (at threshold {eer_thresh:.4f})", flush=True)
        print(f"Optimal Threshold (Youden J): {opt_thresh:.4f} (Max J = {j_scores[opt_idx]:.4f})", flush=True)

        for thresh_name, thresh_val in [("Default 0.5", 0.5), (f"Optimal {opt_thresh:.4f}", opt_thresh)]:
            y_pred = (y_scores >= thresh_val).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            acc = (tp + tn) / len(y_true)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            print(f"\n--- Metrics at {thresh_name} ---", flush=True)
            print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)", flush=True)
            print(f"  Precision: {prec:.4f}", flush=True)
            print(f"  Recall:    {rec:.4f}", flush=True)
            print(f"  F1-Score:  {f1:.4f}", flush=True)
            print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}", flush=True)

        real_probs = y_scores[y_true == 0]
        fake_probs = y_scores[y_true == 1]

        print(f"\n--- Probability Distributions ---", flush=True)
        print(
            f"  Real Videos (N={len(real_probs)}): Mean={real_probs.mean():.4f}, Std={real_probs.std():.4f}, "
            f"Min={real_probs.min():.4f}, Max={real_probs.max():.4f}, Median={np.median(real_probs):.4f}, "
            f"Q25={np.percentile(real_probs, 25):.4f}, Q75={np.percentile(real_probs, 75):.4f}",
            flush=True,
        )
        print(
            f"  Fake Videos (N={len(fake_probs)}): Mean={fake_probs.mean():.4f}, Std={fake_probs.std():.4f}, "
            f"Min={fake_probs.min():.4f}, Max={fake_probs.max():.4f}, Median={np.median(fake_probs):.4f}, "
            f"Q25={np.percentile(fake_probs, 25):.4f}, Q75={np.percentile(fake_probs, 75):.4f}",
            flush=True,
        )

        hash_after = compute_md5(ckpt_path)
        status = "UNCHANGED" if hash_before == hash_after else "MODIFIED!"
        print(f"\nCheckpoint integrity check: {status} (MD5: {hash_before})", flush=True)


if __name__ == "__main__":
    evaluate_all()
