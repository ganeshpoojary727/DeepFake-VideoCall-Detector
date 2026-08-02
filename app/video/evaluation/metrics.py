"""Comprehensive evaluation metrics computation module for video deepfake detection."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import torch


class EvaluationMetrics:
    """Computes production evaluation metrics: Accuracy, Precision, Recall, F1, ROC, AUC, EER."""

    @staticmethod
    def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute classification accuracy."""
        if len(y_true) == 0:
            return 0.0
        return float(np.mean(y_true == y_pred))

    @staticmethod
    def compute_precision_recall_f1(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[float, float, float]:
        """Compute precision, recall, and F1 score for positive class (fake=1)."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return float(precision), float(recall), float(f1)

    @staticmethod
    def compute_roc_auc(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[List[float], List[float], float]:
        """Compute ROC curve (FPR, TPR) and AUC score."""
        if len(y_true) == 0:
            return [], [], 0.0

        thresholds = np.linspace(0.0, 1.0, num=100)
        fpr_list: List[float] = []
        tpr_list: List[float] = []

        num_pos = np.sum(y_true == 1)
        num_neg = np.sum(y_true == 0)

        for th in thresholds:
            y_pred = (y_probs >= th).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            tpr = tp / num_pos if num_pos > 0 else 0.0
            fpr = fp / num_neg if num_neg > 0 else 0.0
            tpr_list.append(float(tpr))
            fpr_list.append(float(fpr))

        # Compute AUC using trapezoidal rule without numpy version incompatibilities
        sorted_indices = np.argsort(fpr_list)
        sorted_fpr = np.array(fpr_list)[sorted_indices]
        sorted_tpr = np.array(tpr_list)[sorted_indices]
        
        # Trapezoidal integration formula: dx * (y1 + y2) / 2
        dx = np.diff(sorted_fpr)
        auc = float(np.sum(dx * (sorted_tpr[:-1] + sorted_tpr[1:]) / 2.0))

        return fpr_list, tpr_list, max(0.0, min(1.0, abs(auc)))

    @classmethod
    def compute_all(cls, y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """Compute all evaluation metrics dictionary."""
        y_pred = (y_probs >= threshold).astype(int)
        acc = cls.compute_accuracy(y_true, y_pred)
        prec, rec, f1 = cls.compute_precision_recall_f1(y_true, y_pred)
        fpr, tpr, auc = cls.compute_roc_auc(y_true, y_probs)

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
            "roc_fpr": fpr,
            "roc_tpr": tpr,
        }
