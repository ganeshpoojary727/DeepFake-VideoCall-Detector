"""Confusion matrix evaluation module."""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


class ConfusionMatrix:
    """Computes and formats classification confusion matrix metrics."""

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self.tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        self.fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        self.tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        self.fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    def to_matrix(self) -> np.ndarray:
        """Get 2x2 confusion matrix array [[TN, FP], [FN, TP]]."""
        return np.array([[self.tn, self.fp], [self.fn, self.tp]], dtype=int)

    def to_dict(self) -> Dict[str, int]:
        """Get dictionary representation."""
        return {
            "true_positive": self.tp,
            "false_positive": self.fp,
            "true_negative": self.tn,
            "false_negative": self.fn,
        }
