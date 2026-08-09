"""Biometric deepfake detection metrics module (EER, HTER, APCER, BPCER)."""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import torch


from app.utils.logger import get_logger

logger = get_logger(__name__)


def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Compute Equal Error Rate (EER) and decision threshold.

    Args:
        bonafide_scores: Scores for bona fide samples (higher score = more likely bonafide).
        spoof_scores: Scores for spoof samples.

    Returns:
        Tuple[Optional[float], Optional[float]]: (eer, threshold) or (None, None) if invalid.
    """
    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return None, None

    if np.isnan(bonafide_scores).any() or np.isnan(spoof_scores).any() or np.isinf(bonafide_scores).any() or np.isinf(spoof_scores).any():
        logger.warning("Non-finite scores detected in compute_eer. Returning INVALID metrics (None).")
        return None, None

    all_scores = np.concatenate([bonafide_scores, spoof_scores])
    thresholds = np.sort(all_scores)

    # FRR (False Rejection Rate of bonafide) and FAR (False Acceptance Rate of spoof)
    frr = np.array([np.mean(bonafide_scores < t) for t in thresholds])
    far = np.array([np.mean(spoof_scores >= t) for t in thresholds])

    # Find threshold where FRR == FAR
    diffs = np.abs(frr - far)
    if np.isnan(diffs).any():
        return None, None
    idx = np.nanargmin(diffs)
    eer = (frr[idx] + far[idx]) / 2.0
    return float(eer), float(thresholds[idx])


def compute_biometric_metrics(
    y_pred_probs: torch.Tensor | np.ndarray,
    y_true: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute biometric error rates APCER, BPCER, HTER, and EER.

    Args:
        y_pred_probs: Model output probabilities for spoof class (1 = Spoof).
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
            "eer_threshold": None,
            "is_valid": False,
        }

    if probs.ndim == 2 and probs.shape[1] == 2:
        spoof_prob = probs[:, 1]
    else:
        spoof_prob = probs

    bonafide_mask = labels == 0
    spoof_mask = labels == 1

    # APCER: Attack Presentation Classification Error Rate (Spoof misclassified as Bonafide)
    # BPCER: Bona Fide Presentation Classification Error Rate (Bonafide misclassified as Spoof)
    apcer = np.mean(spoof_prob[spoof_mask] < threshold) if np.sum(spoof_mask) > 0 else 0.0
    bpcer = np.mean(spoof_prob[bonafide_mask] >= threshold) if np.sum(bonafide_mask) > 0 else 0.0
    hter = (apcer + bpcer) / 2.0

    # EER calculation (bonafide scores = 1 - spoof_prob)
    bonafide_scores = 1.0 - spoof_prob[bonafide_mask]
    spoof_scores = 1.0 - spoof_prob[spoof_mask]
    eer, eer_threshold = compute_eer(bonafide_scores, spoof_scores)

    return {
        "eer": float(eer) if eer is not None else None,
        "hter": float(hter),
        "apcer": float(apcer),
        "bpcer": float(bpcer),
        "eer_threshold": float(eer_threshold) if eer_threshold is not None else None,
        "is_valid": True if eer is not None else False,
    }
