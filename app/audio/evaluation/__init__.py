"""Audio evaluation metrics and benchmark modules."""

from app.audio.evaluation.eer import (
    compute_det_curve,
    compute_eer,
    compute_eer_from_labels,
    compute_min_dcf,
    compute_min_tdcf,
    evaluate_cm_predictions,
)
from app.audio.evaluation.evaluator import AudioEvaluator
from app.audio.evaluation.metrics import MetricsCalculator

__all__ = [
    "compute_eer",
    "compute_eer_from_labels",
    "compute_min_dcf",
    "compute_min_tdcf",
    "compute_det_curve",
    "evaluate_cm_predictions",
    "MetricsCalculator",
    "AudioEvaluator",
]
