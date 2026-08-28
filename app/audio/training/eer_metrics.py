"""Biometric deepfake detection metrics module (EER, min t-DCF, HTER, APCER, BPCER)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch

from app.audio.evaluation.eer import (
    compute_det_curve,
    compute_eer,
    compute_eer_from_labels,
    compute_min_dcf,
    compute_min_tdcf,
    evaluate_cm_predictions,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


def compute_biometric_metrics(
    y_pred_probs: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute biometric error rates APCER, BPCER, HTER, EER, and min t-DCF.

    Args:
        y_pred_probs: Model output probabilities for spoof class (1 = Spoof) or 2-class softmax.
        y_true: Ground truth binary labels (0 = Bonafide, 1 = Spoof).
        threshold: Operating decision threshold.

    Returns:
        Dict[str, Any]: Biometric metrics dictionary.
    """
    if isinstance(y_pred_probs, torch.Tensor):
        probs = y_pred_probs.detach().cpu().numpy()
    else:
        probs = np.array(y_pred_probs)

    if isinstance(y_true, torch.Tensor):
        labels = y_true.detach().cpu().numpy()
    else:
        labels = np.array(y_true)

    if np.isnan(probs).any() or np.isinf(probs).any():
        logger.warning("Non-finite probabilities detected in compute_biometric_metrics. Reporting INVALID.")
        return {
            "eer": None,
            "hter": None,
            "apcer": None,
            "bpcer": None,
            "min_tdcf": None,
            "min_dcf": None,
            "eer_threshold": None,
            "is_valid": False,
        }

    metrics = evaluate_cm_predictions(y_true=labels, y_scores=probs, threshold=threshold)
    metrics["is_valid"] = True if metrics.get("eer") is not None else False
    return metrics


__all__ = [
    "compute_eer",
    "compute_eer_from_labels",
    "compute_min_dcf",
    "compute_min_tdcf",
    "compute_det_curve",
    "evaluate_cm_predictions",
    "compute_biometric_metrics",
]
