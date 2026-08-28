"""
Model evaluation with production-grade metrics.

Improvements over v1
─────────────────────
• Returns ``EvaluationResult`` dataclass instead of fragile 6-element tuple
• Adds **Equal Error Rate (EER)** — the standard anti-spoofing metric
• Collects prediction probabilities for threshold analysis
• Consistent 4-space indentation
• Structured logging
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from app.core.interfaces import EvaluationResult
from app.utils.logger import get_logger

logger = get_logger(__name__)


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute the Equal Error Rate (EER).

    EER is the point where False Accept Rate == False Reject Rate.
    This is the standard metric for anti-spoofing (ASVspoof).

    Parameters
    ----------
    labels : np.ndarray
        Ground truth binary labels (0 = bonafide, 1 = spoof).
    scores : np.ndarray
        Prediction scores / probabilities for the spoof class.

    Returns
    -------
    float
        The EER value (0.0 – 1.0).  Lower is better.
    """
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    except ValueError:
        # Fallback: find closest point
        fnr = 1 - tpr
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = float(fpr[idx])
    return eer


class Evaluator:
    """
    Evaluate a trained model on a test dataset.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    test_loader : DataLoader
        Test dataset loader.
    device : torch.device
        Compute device.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.model = model
        self.test_loader = test_loader
        self.device = device

    def evaluate(self) -> EvaluationResult:
        """
        Run evaluation and return structured metrics.

        Returns
        -------
        EvaluationResult
            Contains accuracy, precision, recall, F1, EER, confusion matrix,
            and classification report.
        """
        self.model.eval()

        all_predictions: List[int] = []
        all_labels: List[int] = []
        all_spoof_scores: List[float] = []  # for EER computation

        with torch.no_grad():
            for features, labels in self.test_loader:
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(features)

                # Probabilities for EER
                probs = F.softmax(outputs, dim=1)
                spoof_probs = probs[:, 1]  # probability of spoof class
                all_spoof_scores.extend(spoof_probs.cpu().numpy().tolist())

                # Hard predictions
                predicted = outputs.argmax(dim=1)
                all_predictions.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        labels_np = np.array(all_labels)
        preds_np = np.array(all_predictions)
        scores_np = np.array(all_spoof_scores)

        # Standard metrics
        accuracy = accuracy_score(labels_np, preds_np)
        precision = precision_score(labels_np, preds_np, zero_division=0)
        recall = recall_score(labels_np, preds_np, zero_division=0)
        f1 = f1_score(labels_np, preds_np, zero_division=0)
        matrix = confusion_matrix(labels_np, preds_np)
        report = classification_report(
            labels_np, preds_np, target_names=["bonafide", "spoof"]
        )

        # EER (the key anti-spoofing metric)
        eer = None
        try:
            eer = compute_eer(labels_np, scores_np)
            logger.info("EER: %.4f (%.2f%%)", eer, eer * 100)
        except Exception as exc:
            logger.warning("Could not compute EER: %s", exc)

        result = EvaluationResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            eer=eer,
            confusion_matrix=matrix,
            classification_report=report,
        )

        logger.info("Accuracy:  %.4f (%.2f%%)", accuracy, accuracy * 100)
        logger.info("Precision: %.4f", precision)
        logger.info("Recall:    %.4f", recall)
        logger.info("F1 Score:  %.4f", f1)

        return result


# Alias for backward compatibility
AudioEvaluator = Evaluator