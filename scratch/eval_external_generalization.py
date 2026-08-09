"""EXTERNAL GENERALIZATION / ZERO-SHOT EVALUATION

Evaluates frozen AASIST model (trained_models/audio/best_model.pt) on the unseen
ASVspoof2019 LA Evaluation Dataset (71,237 samples) without fine-tuning or retraining.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from app.audio.models.aasist import AASIST
from app.audio.datasets.dataloader import create_test_dataloader
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

    print("=" * 60)
    print("EXTERNAL GENERALIZATION / ZERO-SHOT EVALUATION")
    print("=" * 60)
    print(f"Model Checkpoint Path: {ckpt_path}")
    print(f"Pre-evaluation MD5:   {md5_before}")
    print(f"Pre-evaluation Size:  {size_before} bytes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device:   {device}")

    # 2. Load frozen model with strict=True
    model = AASIST(num_classes=2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    epoch_saved = ckpt.get("epoch")

    print(f"Saved Model Epoch:   {epoch_saved}")
    model.load_state_dict(state_dict, strict=True)
    print("Model State Dict:    Loaded successfully with strict=True.")
    model.eval()

    # 3. Create unseen external test dataloader
    test_loader = create_test_dataloader()
    dataset = test_loader.dataset
    total_samples = len(dataset.samples)

    num_bonafide = sum(1 for _, label in dataset.samples if label == 0)
    num_spoof = sum(1 for _, label in dataset.samples if label == 1)

    print(f"Dataset Name:        ASVspoof2019 LA Evaluation (Unseen)")
    print(f"Total Dataset Size:  {total_samples} samples")
    print(f"Bona Fide Count:     {num_bonafide}")
    print(f"Spoof Count:         {num_spoof}")

    all_logits = []
    all_labels = []
    non_finite_logit_count = 0
    skipped_samples_count = 0
    total_eval_time_ms = 0.0

    # 4. Read-Only Zero-Shot Evaluation Loop
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Zero-Shot External Evaluation"):
            x = batch["tensor"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            t0 = time.perf_counter()
            with torch.amp.autocast("cuda"):
                logits = model(x)
            t1 = time.perf_counter()
            total_eval_time_ms += (t1 - t0) * 1000.0

            # Non-finite check
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                non_finite_logit_count += logits.size(0)
                skipped_samples_count += logits.size(0)
                continue

            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())

    total_samples_evaluated = sum(l.size(0) for l in all_logits)

    if total_samples_evaluated == 0:
        print("ERROR: Zero valid samples evaluated!")
        return

    cat_logits = torch.cat(all_logits, dim=0)
    cat_labels = torch.cat(all_labels, dim=0)

    # 5. Calculate logit & probability stats
    logit_mean = cat_logits.mean().item()
    logit_min = cat_logits.min().item()
    logit_max = cat_logits.max().item()

    probs = torch.softmax(cat_logits, dim=-1)
    bonafide_probs = probs[:, 0]  # Class 0 = Bonafide
    bf_prob_mean = bonafide_probs.mean().item()
    bf_prob_min = bonafide_probs.min().item()
    bf_prob_max = bonafide_probs.max().item()

    # 6. Calculate metrics using standard AudioMetricsCalculator
    avg_latency_ms = total_eval_time_ms / total_samples_evaluated
    full_metrics = AudioMetricsCalculator.compute_all(cat_logits, cat_labels, latency_ms=avg_latency_ms)

    # 7. Post-eval hash verification
    data_after = ckpt_path.read_bytes()
    md5_after = hashlib.md5(data_after).hexdigest()
    size_after = len(data_after)
    is_unchanged = (md5_before == md5_after) and (size_before == size_after)

    # 8. Report Results
    report = {
        "experiment_type": "EXTERNAL GENERALIZATION / ZERO-SHOT EVALUATION",
        "checkpoint_path": str(ckpt_path),
        "dataset_name": "ASVspoof2019 LA Evaluation (Unseen)",
        "total_samples": total_samples_evaluated,
        "bona_fide_samples": num_bonafide,
        "spoof_samples": num_spoof,
        "accuracy": full_metrics.get("accuracy"),
        "precision": full_metrics.get("precision"),
        "recall": full_metrics.get("recall"),
        "f1_score": full_metrics.get("f1"),
        "eer": full_metrics.get("eer"),
        "hter": full_metrics.get("hter"),
        "apcer": full_metrics.get("apcer"),
        "bpcer": full_metrics.get("bpcer"),
        "eer_threshold": full_metrics.get("eer_threshold"),
        "confusion_matrix": full_metrics.get("confusion_matrix"),
        "non_finite_logits": non_finite_logit_count,
        "skipped_samples": skipped_samples_count,
        "inference_latency_ms_per_sample": avg_latency_ms,
        "logit_stats": {"mean": logit_mean, "min": logit_min, "max": logit_max},
        "bonafide_prob_stats": {"mean": bf_prob_mean, "min": bf_prob_min, "max": bf_prob_max},
        "checkpoint_integrity_passed": is_unchanged,
        "md5_before": md5_before,
        "md5_after": md5_after,
    }

    report_path = Path("logs/external_generalization_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("EXTERNAL GENERALIZATION REPORT SUMMARY")
    print("=" * 60)
    print(f"1. Total Samples Evaluated:            {total_samples_evaluated}")
    print(f"2. Bona Fide Samples:                 {num_bonafide}")
    print(f"3. Spoof Samples:                     {num_spoof}")
    print(f"4. Accuracy:                          {full_metrics['accuracy']:.4%}" if full_metrics['accuracy'] is not None else "Accuracy: INVALID")
    print(f"5. Precision:                         {full_metrics['precision']:.4f}" if full_metrics['precision'] is not None else "Precision: INVALID")
    print(f"6. Recall:                            {full_metrics['recall']:.4f}" if full_metrics['recall'] is not None else "Recall: INVALID")
    print(f"7. F1 Score:                          {full_metrics['f1']:.4f}" if full_metrics['f1'] is not None else "F1: INVALID")
    print(f"8. Equal Error Rate (EER):            {full_metrics['eer']:.4%}" if full_metrics['eer'] is not None else "EER: INVALID")
    print(f"9. Half-Total Error Rate (HTER):      {full_metrics['hter']:.4%}" if full_metrics['hter'] is not None else "HTER: INVALID")
    print(f"10. Confusion Matrix:                 {full_metrics['confusion_matrix']}")
    print(f"11. Non-finite Logits Count:          {non_finite_logit_count}")
    print(f"12. Skipped Samples Count:             {skipped_samples_count}")
    print(f"13. Inference Latency (ms/sample):     {avg_latency_ms:.3f} ms")
    print(f"14. Logit Stats (Mean/Min/Max):        Mean={logit_mean:.4f}, Min={logit_min:.4f}, Max={logit_max:.4f}")
    print(f"15. Bonafide Prob (Mean/Min/Max):     Mean={bf_prob_mean:.4f}, Min={bf_prob_min:.4f}, Max={bf_prob_max:.4f}")
    print("-" * 60)
    print(f"Checkpoint Unchanged Verification:     {'PASSED' if is_unchanged else 'FAILED'}")
    print(f"Report saved to:                       {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
