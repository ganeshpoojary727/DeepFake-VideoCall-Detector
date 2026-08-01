"""Biometric deepfake detection metrics module (EER, HTER, APCER, BPCER)."""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import torch


def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER) and decision threshold.

    Args:
        bonafide_scores: Scores for bona fide samples (higher score = more likely bonafide).
        spoof_scores: Scores for spoof samples.

    Returns:
        Tuple[float, float]: (eer, threshold).
    """
    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return 0.0, 0.5

    all_scores = np.concatenate([bonafide_scores, spoof_scores])
    thresholds = np.sort(all_scores)

    # FRR (False Rejection Rate of bonafide) and FAR (False Acceptance Rate of spoof)
    frr = np.array([np.mean(bonafide_scores < t) for t in thresholds])
    far = np.array([np.mean(spoof_scores >= t) for t in thresholds])

    # Find threshold where FRR == FAR
    idx = np.nanargmin(np.abs(frr - far))
    eer = (frr[idx] + far[idx]) / 2.0
    return float(eer), float(thresholds[idx])


def compute_biometric_metrics(
    y_pred_probs: torch.Tensor | np.ndarray,
    y_true: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute biometric error rates APCER, BPCER, HTER, and EER.

    Args:
        y_pred_probs: Model output probabilities for spoof class (1 = Spoof).
        y_true: Ground truth binary labels (0 = Bonafide, 1 = Spoof).
        threshold: Operating decision threshold.

    Returns:
        Dict[str, float]: Biometric metrics dictionary.
    """
    if isinstance(y_pred_probs, torch.Tensor):
        probs = y_pred_probs.detach().cpu().numpy()
    else:
        probs = np.array(y_pred_probs)

    if isinstance(y_true, torch.Tensor):
        labels = y_true.detach().cpu().numpy()
    else:
        labels = np.array(y_true)

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
        "eer": float(eer),
        "hter": float(hter),
        "apcer": float(apcer),
        "bpcer": float(bpcer),
        "eer_threshold": float(eer_threshold),
    }
