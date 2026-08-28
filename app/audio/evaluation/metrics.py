"""
Evaluation metrics calculation module for audio deepfake classification.

Provides the MetricsCalculator class for computing Accuracy, Precision, Recall,
F1-Score, Confusion Matrix, Equal Error Rate (EER), and min t-DCF / minDCF.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from app.audio.evaluation.eer import compute_eer_from_labels, compute_min_dcf
from app.audio.utils.logger import AudioLogger

logger = AudioLogger.get("evaluation.metrics")


class MetricsCalculator:
    """
    Calculator for standard classification and biometrics evaluation metrics (EER, minDCF).
    """

    @staticmethod
    def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
        """
        Compute Equal Error Rate (EER) and the corresponding optimal threshold.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels (0 = bonafide, 1 = spoof).
        y_scores : np.ndarray
            Model predicted probability or logit scores for spoof class.

        Returns
        -------
        Tuple[float, float]
            (eer, eer_threshold) tuple.
        """
        return compute_eer_from_labels(y_true, y_scores)

    @staticmethod
    def compute_min_dcf(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        p_target: float = 0.05,
        c_miss: float = 1.0,
        c_fa: float = 1.0,
    ) -> float:
        """
        Compute normalized Minimum Detection Cost Function (minDCF).

        Parameters
        ----------
        p_target : float
            Prior probability of target spoof attack.
        c_miss : float
            Cost of a miss (false acceptance).
        c_fa : float
            Cost of a false alarm (false rejection).

        Returns
        -------
        float
            Normalized minDCF value.
        """
        return compute_min_dcf(
            y_true=y_true,
            y_scores=y_scores,
            p_target=p_target,
            c_miss=c_miss,
            c_fa=c_fa,
        )

    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Compute complete metric suite for classification and biometrics evaluation.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels array.
        y_pred : np.ndarray
            Predicted hard class labels array.
        y_scores : Optional[np.ndarray]
            Predicted spoof class probabilities array.

        Returns
        -------
        Dict[str, Union[float, np.ndarray]]
            Dictionary of calculated metrics.
        """
        acc = float(accuracy_score(y_true, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred)

        results: Dict[str, Union[float, np.ndarray]] = {
            "accuracy": round(acc, 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "confusion_matrix": cm,
        }

        if y_scores is not None and len(np.unique(y_true)) > 1:
            try:
                eer, threshold = self.compute_eer(y_true, y_scores)
                min_dcf = self.compute_min_dcf(y_true, y_scores)
                results["eer"] = round(eer, 4)
                results["eer_threshold"] = round(threshold, 4)
                results["min_dcf"] = round(min_dcf, 4)
                results["min_tdcf"] = round(min_dcf, 4)
            except Exception as err:
                logger.warning("Could not calculate EER/minDCF: %s", err)
                results["eer"] = 0.0

        return results
