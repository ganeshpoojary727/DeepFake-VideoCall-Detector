"""Evaluation script for final best_model.pt on ASVspoof2019 validation dataset."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from app.audio.models.aasist import AASIST
from app.audio.datasets.dataloader import create_validation_dataloader
from app.audio.training.metrics import AudioMetricsCalculator
from app.audio.training.eer_metrics import compute_biometric_metrics
from app.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    ckpt_path = Path("trained_models/audio/best_model.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    # 1. Record pre-eval hash
    data_before = ckpt_path.read_bytes()
    md5_before = hashlib.md5(data_before).hexdigest()
    size_before = len(data_before)

    print(f"Loaded checkpoint from: {ckpt_path}")
    print(f"Pre-evaluation MD5: {md5_before}")
    print(f"Pre-evaluation Size: {size_before} bytes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")

    # 2. Load model
    model = AASIST(num_classes=2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    epoch = ckpt.get("epoch")
    metrics_saved = ckpt.get("metrics")

    print(f"Saved Checkpoint Epoch: {epoch}")
    print(f"Saved Checkpoint Metrics: {metrics_saved}")

    # Load state_dict (using strict=True)
    model.load_state_dict(state_dict, strict=True)
    print("State dict loaded successfully with strict=True.")

    model.eval()

    # 3. Create validation dataloader (same pipeline as training)
    val_loader = create_validation_dataloader()
    dataset = val_loader.dataset
    print(f"Validation dataset total samples: {len(dataset)}")

    num_bonafide = sum(1 for _, label in dataset.samples if label == 0)
    num_spoof = sum(1 for _, label in dataset.samples if label == 1)

    print(f"Dataset stats: Bonafide={num_bonafide}, Spoof={num_spoof}, Total={len(dataset.samples)}")

    all_logits = []
    all_labels = []
    non_finite_logit_count = 0
    skipped_samples_count = 0
    total_eval_time_ms = 0.0

    # 4. Evaluation Loop
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating best_model.pt"):
            x = batch["tensor"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            t0 = time.perf_counter()
            with torch.amp.autocast("cuda"):
                logits = model(x)
            t1 = time.perf_counter()
            total_eval_time_ms += (t1 - t0) * 1000.0

            # Check finite
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                non_finite_logit_count += logits.size(0)
                skipped_samples_count += logits.size(0)
                continue

            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())

    total_samples_evaluated = sum(l.size(0) for l in all_logits)

    if total_samples_evaluated == 0:
        print("ERROR: No valid samples evaluated!")
        return

    cat_logits = torch.cat(all_logits, dim=0)
    cat_labels = torch.cat(all_labels, dim=0)

    # Calculate logit stats
    logit_mean = cat_logits.mean().item()
    logit_min = cat_logits.min().item()
    logit_max = cat_logits.max().item()

    # Probabilities & Bonafide probability stats
    probs = torch.softmax(cat_logits, dim=-1)
    bonafide_probs = probs[:, 0]  # Label 0 = Bonafide
    bf_prob_mean = bonafide_probs.mean().item()
    bf_prob_min = bonafide_probs.min().item()
    bf_prob_max = bonafide_probs.max().item()

    # Metrics calculation
    avg_latency_ms = total_eval_time_ms / total_samples_evaluated
    full_metrics = AudioMetricsCalculator.compute_all(cat_logits, cat_labels, latency_ms=avg_latency_ms)

    # Post-eval hash check
    data_after = ckpt_path.read_bytes()
    md5_after = hashlib.md5(data_after).hexdigest()
    size_after = len(data_after)
    is_unchanged = (md5_before == md5_after) and (size_before == size_after)

    print("\n" + "=" * 50)
    print("ASVSPOOF2019 VALIDATION EVALUATION REPORT")
    print("=" * 50)
    print(f"1. Bona fide samples count:            {num_bonafide}")
    print(f"2. Spoof samples count:                {num_spoof}")
    print(f"3. Total samples evaluated:            {total_samples_evaluated}")
    print(f"4. Accuracy:                           {full_metrics['accuracy']:.4%}" if full_metrics['accuracy'] is not None else "Accuracy: INVALID")
    print(f"5. Precision:                          {full_metrics['precision']:.4f}" if full_metrics['precision'] is not None else "Precision: INVALID")
    print(f"6. Recall:                             {full_metrics['recall']:.4f}" if full_metrics['recall'] is not None else "Recall: INVALID")
    print(f"7. F1 Score:                           {full_metrics['f1']:.4f}" if full_metrics['f1'] is not None else "F1: INVALID")
    print(f"8. Equal Error Rate (EER):             {full_metrics['eer']:.4%}" if full_metrics['eer'] is not None else "EER: INVALID")
    print(f"9. Half-Total Error Rate (HTER):       {full_metrics['hter']:.4%}" if full_metrics['hter'] is not None else "HTER: INVALID")
    print(f"10. Confusion Matrix:                  {full_metrics['confusion_matrix']}")
    print(f"11. Samples with non-finite logits:   {non_finite_logit_count}")
    print(f"12. Samples skipped:                   {skipped_samples_count}")
    print(f"13. Logits stats (Mean/Min/Max):       Mean={logit_mean:.4f}, Min={logit_min:.4f}, Max={logit_max:.4f}")
    print(f"14. Bonafide prob (Mean/Min/Max):     Mean={bf_prob_mean:.4f}, Min={bf_prob_min:.4f}, Max={bf_prob_max:.4f}")
    print("-" * 50)
    print(f"Checkpoint Unchanged Verification:      {'PASSED' if is_unchanged else 'FAILED'}")
    print(f"Post-eval MD5:                          {md5_after}")
    print("=" * 50)

if __name__ == "__main__":
    main()
