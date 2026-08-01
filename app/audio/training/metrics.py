"""Comprehensive metric calculation module for AASIST audio model training and evaluation."""

from __future__ import annotations

import time
from typing import Dict, List, Optional
import numpy as np
import torch

from app.audio.training.eer_metrics import compute_biometric_metrics


class AudioMetricsCalculator:
    """Calculates accuracy, precision, recall, F1, ROC AUC, confusion matrix, EER, HTER, APCER, BPCER, latency, and GPU usage."""

    @staticmethod
    def compute_all(
        y_pred_logits: torch.Tensor,
        y_true: torch.Tensor,
        latency_ms: float = 0.0,
    ) -> Dict[str, Any]:
        """Compute complete audio deepfake evaluation metrics suite.

        Args:
            y_pred_logits: Raw model classification logits [B, num_classes].
            y_true: Ground truth binary labels [B].
            latency_ms: Measured inference latency per sample.

        Returns:
            Dict[str, Any]: Comprehensive metrics dictionary.
        """
        if y_pred_logits.numel() == 0:
            return {"accuracy": 0.0, "f1": 0.0, "eer": 0.0}

        probs = torch.softmax(y_pred_logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        # Accuracy
        correct = (preds == y_true).float().sum().item()
        accuracy = correct / y_true.numel()

        # Confusion Matrix
        tp = ((preds == 1) & (y_true == 1)).float().sum().item()
        fp = ((preds == 1) & (y_true == 0)).float().sum().item()
        fn = ((preds == 0) & (y_true == 1)).float().sum().item()
        tn = ((preds == 0) & (y_true == 0)).float().sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Biometric error rates (EER, HTER, APCER, BPCER)
        bio = compute_biometric_metrics(probs, y_true)

        # GPU Utilization / Memory (if CUDA available)
        gpu_memory_mb = 0.0
        if torch.cuda.is_available():
            gpu_memory_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
            "latency_ms": float(latency_ms),
            "gpu_memory_mb": float(gpu_memory_mb),
            **bio,
        }
