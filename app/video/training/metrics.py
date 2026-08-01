"""Video metrics calculation module."""

from __future__ import annotations

from typing import Dict, List
import numpy as np
import torch


class VideoMetricsCalculator:
    """Calculates accuracy, precision, recall, F1, and log loss for video deepfake detection."""

    @staticmethod
    def compute_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """Compute classification accuracy ratio."""
        if y_pred.numel() == 0:
            return 0.0
        preds = torch.argmax(y_pred, dim=-1) if y_pred.dim() > 1 else (y_pred > 0.5).long()
        correct = (preds == y_true).float().sum().item()
        return correct / y_true.numel()

    @staticmethod
    def compute_precision_recall_f1(y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
        """Compute precision, recall, and F1 score."""
        if y_pred.numel() == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        preds = torch.argmax(y_pred, dim=-1) if y_pred.dim() > 1 else (y_pred > 0.5).long()
        tp = ((preds == 1) & (y_true == 1)).float().sum().item()
        fp = ((preds == 1) & (y_true == 0)).float().sum().item()
        fn = ((preds == 0) & (y_true == 1)).float().sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}

    @classmethod
    def compute_all(cls, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
        """Compute all video classification metrics."""
        acc = cls.compute_accuracy(y_pred, y_true)
        prf = cls.compute_precision_recall_f1(y_pred, y_true)
        return {
            "accuracy": acc,
            **prf,
        }


# Convenience function alias
def calculate_video_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
    """Calculate accuracy and F1 metrics dictionary."""
    return VideoMetricsCalculator.compute_all(y_pred, y_true)
