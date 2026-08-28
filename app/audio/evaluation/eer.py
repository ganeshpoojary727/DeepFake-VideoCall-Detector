"""
Biometric Deepfake & Countermeasure Evaluation Metrics module.

Provides calibrated calculation of Equal Error Rate (EER), minimum tandem
Detection Cost Function (min t-DCF / min DCF), DET curve generation, and
comprehensive classifier benchmarking for ASVspoof 2019/2021 evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from app.audio.constants.audio_constants import (
    DEFAULT_MIN_DCF_C_FA,
    DEFAULT_MIN_DCF_C_MISS,
    DEFAULT_MIN_DCF_P_TARGET,
    LABEL_BONAFIDE,
    LABEL_SPOOF,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


def compute_eer(
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute Equal Error Rate (EER) and operating threshold from raw score partitions.

    Convention: Higher score means MORE LIKELY BONAFIDE (genuine).
    - False Rejection Rate (FRR): fraction of bona fide scores < threshold
    - False Acceptance Rate (FAR): fraction of spoof scores >= threshold

    Parameters
    ----------
    bonafide_scores : np.ndarray
        Array of scores for genuine (bonafide) utterances.
    spoof_scores : np.ndarray
        Array of scores for synthetic/spoofed utterances.

    Returns
    -------
    Tuple[Optional[float], Optional[float]]
        (eer_rate, threshold_val) in [0.0, 1.0], or (None, None) if inputs are invalid.
    """
    bonafide_scores = np.asarray(bonafide_scores, dtype=np.float64).ravel()
    spoof_scores = np.asarray(spoof_scores, dtype=np.float64).ravel()

    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        logger.warning("compute_eer: Empty bona fide or spoof score partition.")
        return None, None

    if not np.isfinite(bonafide_scores).all() or not np.isfinite(spoof_scores).all():
        logger.warning("compute_eer: Non-finite values detected in score partitions.")
        return None, None

    # Concatenate and sort unique candidate thresholds
    all_scores = np.sort(np.unique(np.concatenate([bonafide_scores, spoof_scores])))
    if len(all_scores) == 1:
        return 0.5, float(all_scores[0])

    n_bona = len(bonafide_scores)
    n_spoof = len(spoof_scores)

    # Efficient sorted search for FRR & FAR
    sorted_bona = np.sort(bonafide_scores)
    sorted_spoof = np.sort(spoof_scores)

    # FRR: fraction of bona fide < threshold
    frr = np.searchsorted(sorted_bona, all_scores, side="left") / float(n_bona)
    # FAR: fraction of spoof >= threshold
    far = (n_spoof - np.searchsorted(sorted_spoof, all_scores, side="left")) / float(n_spoof)

    # Find crossing point |FRR - FAR|
    diff = frr - far
    zero_crossings = np.where(np.diff(np.sign(diff)))[0]

    if len(zero_crossings) > 0:
        idx = zero_crossings[0]
        # Linear interpolation between idx and idx + 1
        x1, x2 = all_scores[idx], all_scores[idx + 1]
        frr1, frr2 = frr[idx], frr[idx + 1]
        far1, far2 = far[idx], far[idx + 1]

        # Solve (frr1 + (frr2 - frr1)*t) = (far1 + (far2 - far1)*t)
        denom = (far2 - far1) - (frr2 - frr1)
        if abs(denom) > 1e-12:
            t = (frr1 - far1) / denom
            t = np.clip(t, 0.0, 1.0)
            threshold = x1 + t * (x2 - x1)
            eer = frr1 + t * (frr2 - frr1)
        else:
            threshold = (x1 + x2) / 2.0
            eer = (frr1 + far1) / 2.0
    else:
        min_idx = np.argmin(np.abs(diff))
        threshold = all_scores[min_idx]
        eer = (frr[min_idx] + far[min_idx]) / 2.0

    return float(np.clip(eer, 0.0, 1.0)), float(threshold)


def compute_eer_from_labels(
    y_true: Union[np.ndarray, torch.Tensor],
    y_scores: Union[np.ndarray, torch.Tensor],
) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER) given ground truth binary labels and spoof probabilities/scores.

    Parameters
    ----------
    y_true : np.ndarray | torch.Tensor
        Ground truth labels (0 = bonafide, 1 = spoof).
    y_scores : np.ndarray | torch.Tensor
        Predicted probability / logit that the sample is SPOOF.

    Returns
    -------
    Tuple[float, float]
        (eer_rate, threshold_for_spoof_scores)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()

    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    y_scores = np.asarray(y_scores, dtype=np.float64).ravel()

    bonafide_mask = (y_true == LABEL_BONAFIDE)
    spoof_mask = (y_true == LABEL_SPOOF)

    if np.sum(bonafide_mask) == 0 or np.sum(spoof_mask) == 0:
        return 0.0, 0.5

    # In ASVspoof, bonafide_scores are higher for genuine utterances (e.g. 1.0 - spoof_prob)
    bonafide_scores = 1.0 - y_scores[bonafide_mask]
    spoof_scores = 1.0 - y_scores[spoof_mask]

    eer, bona_threshold = compute_eer(bonafide_scores, spoof_scores)
    if eer is None or bona_threshold is None:
        return 0.0, 0.5

    # Convert threshold back to spoof probability space: theta_spoof = 1.0 - theta_bona
    spoof_threshold = 1.0 - bona_threshold
    return float(eer), float(spoof_threshold)


def compute_min_dcf(
    y_true: Union[np.ndarray, torch.Tensor],
    y_scores: Union[np.ndarray, torch.Tensor],
    p_target: float = DEFAULT_MIN_DCF_P_TARGET,
    c_miss: float = DEFAULT_MIN_DCF_C_MISS,
    c_fa: float = DEFAULT_MIN_DCF_C_FA,
) -> float:
    """Compute normalized minimum Detection Cost Function (minDCF / min t-DCF proxy).

    Parameters
    ----------
    y_true : np.ndarray | torch.Tensor
        Ground truth binary labels (0 = bonafide, 1 = spoof).
    y_scores : np.ndarray | torch.Tensor
        Predicted spoof probabilities in [0.0, 1.0].
    p_target : float
        Prior probability of spoof attack. Default: 0.05.
    c_miss : float
        Cost of miss (spoof classified as bona fide). Default: 1.0.
    c_fa : float
        Cost of false alarm (bona fide classified as spoof). Default: 1.0.

    Returns
    -------
    float
        Normalized minimum detection cost function value.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()

    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    y_scores = np.asarray(y_scores, dtype=np.float64).ravel()

    if len(np.unique(y_true)) < 2:
        return 0.0

    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=LABEL_SPOOF)
    fnr = 1.0 - tpr  # Miss rate (spoof classified as bonafide)

    # Raw detection cost
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1.0 - p_target)
    # Default uninformative cost
    c_default = min(c_miss * p_target, c_fa * (1.0 - p_target))

    if c_default == 0:
        return 0.0

    min_dcf = float(np.min(dcf) / c_default)
    return float(np.clip(min_dcf, 0.0, 1.0))


def compute_min_tdcf(
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
    p_target: float = DEFAULT_MIN_DCF_P_TARGET,
    c_miss: float = DEFAULT_MIN_DCF_C_MISS,
    c_fa: float = DEFAULT_MIN_DCF_C_FA,
) -> float:
    """Compute ASVspoof normalized minimum tandem Detection Cost Function (min t-DCF).

    Parameters
    ----------
    bonafide_scores : np.ndarray
        Bonafide sample score array (higher = more bonafide).
    spoof_scores : np.ndarray
        Spoof sample score array (higher = more bonafide).
    p_target : float
        Prior probability of spoof attack.
    c_miss : float
        Cost of miss.
    c_fa : float
        Cost of false alarm.

    Returns
    -------
    float
        Normalized min t-DCF value.
    """
    bonafide_scores = np.asarray(bonafide_scores, dtype=np.float64).ravel()
    spoof_scores = np.asarray(spoof_scores, dtype=np.float64).ravel()

    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return 0.0

    # Build labels: bonafide=0, spoof=1
    y_true = np.concatenate([
        np.zeros(len(bonafide_scores), dtype=np.int32),
        np.ones(len(spoof_scores), dtype=np.int32),
    ])
    # Convert bonafide scores to spoof probabilities/scores: spoof_score = -bona_score
    y_scores = np.concatenate([
        -bonafide_scores,
        -spoof_scores,
    ])

    return compute_min_dcf(y_true, y_scores, p_target=p_target, c_miss=c_miss, c_fa=c_fa)


def compute_det_curve(
    y_true: Union[np.ndarray, torch.Tensor],
    y_scores: Union[np.ndarray, torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Detection Error Tradeoff (DET) curve arrays (FPR/FAR, FNR/FRR, thresholds).

    Parameters
    ----------
    y_true : np.ndarray | torch.Tensor
        Ground truth binary labels (0 = bonafide, 1 = spoof).
    y_scores : np.ndarray | torch.Tensor
        Predicted spoof probabilities.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (fpr_array, fnr_array, thresholds)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()

    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    y_scores = np.asarray(y_scores, dtype=np.float64).ravel()

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=LABEL_SPOOF)
    fnr = 1.0 - tpr
    return fpr, fnr, thresholds


def evaluate_cm_predictions(
    y_true: Union[np.ndarray, torch.Tensor],
    y_scores: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculate comprehensive biometric countermeasure evaluation metrics.

    Calculates:
    - EER (Equal Error Rate) and EER decision threshold
    - min t-DCF / min DCF (normalized minimum Detection Cost Function)
    - AUC-ROC (Area Under the ROC Curve)
    - Accuracy, Precision, Recall, F1-Score
    - APCER (Attack Presentation Classification Error Rate)
    - BPCER (Bona Fide Presentation Classification Error Rate)
    - HTER (Half Total Error Rate)
    - Confusion Matrix

    Parameters
    ----------
    y_true : np.ndarray | torch.Tensor
        Ground truth binary labels (0 = bonafide, 1 = spoof).
    y_scores : np.ndarray | torch.Tensor
        Predicted spoof probabilities or 2-class softmax logits (batch, 2).
    threshold : float
        Operating decision threshold (default: 0.5).

    Returns
    -------
    Dict[str, Any]
        Standardized dictionary of evaluation metrics.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()

    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    scores = np.asarray(y_scores, dtype=np.float64)

    if scores.ndim == 2 and scores.shape[1] == 2:
        spoof_prob = scores[:, 1]
    else:
        spoof_prob = scores.ravel()

    # Hard predictions
    y_pred = (spoof_prob >= threshold).astype(np.int32)

    # Basic classification metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    # Biometric error rates
    bonafide_mask = (y_true == LABEL_BONAFIDE)
    spoof_mask = (y_true == LABEL_SPOOF)

    apcer = float(np.mean(spoof_prob[spoof_mask] < threshold)) if np.sum(spoof_mask) > 0 else 0.0
    bpcer = float(np.mean(spoof_prob[bonafide_mask] >= threshold)) if np.sum(bonafide_mask) > 0 else 0.0
    hter = float((apcer + bpcer) / 2.0)

    # EER & min DCF
    eer, eer_threshold = compute_eer_from_labels(y_true, spoof_prob)
    min_dcf = compute_min_dcf(y_true, spoof_prob)

    # AUC-ROC
    try:
        auc = float(roc_auc_score(y_true, spoof_prob)) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        auc = 0.5

    return {
        "eer": round(eer, 4),
        "eer_threshold": round(eer_threshold, 4),
        "min_tdcf": round(min_dcf, 4),
        "min_dcf": round(min_dcf, 4),
        "auc": round(auc, 4),
        "accuracy": round(acc, 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "apcer": round(apcer, 4),
        "bpcer": round(bpcer, 4),
        "hter": round(hter, 4),
        "confusion_matrix": cm,
    }
