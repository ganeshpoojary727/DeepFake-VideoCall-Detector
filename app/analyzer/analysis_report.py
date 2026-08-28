"""
Unified analysis report for deepfake detection results with dual-sided Real/Fake confidence.

This module provides a single structured result type used by all analyzers
(image, video, audio) for consistent output across the entire system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AnalysisReport:
    """Unified result returned by any media analyzer.

    Attributes
    ----------
    verdict : str
        Three-way classification: ``"REAL"``, ``"FAKE"``, or ``"UNCERTAIN"``.
    confidence : float
        Confidence associated with the verdict, in ``[0.0, 1.0]``.
    media_type : str
        Type of media analyzed: ``"image"``, ``"video"``, or ``"audio"``.
    real_confidence : float
        Calibrated probability that the media is authentic/real, in ``[0.0, 1.0]``.
    fake_confidence : float
        Calibrated probability that the media is synthetic/fake, in ``[0.0, 1.0]``.
    scores : dict
        Per-modality fake probabilities, e.g.
        ``{"audio": 0.12, "video": 0.87, "fused": 0.57}``.
        Missing modalities are ``None``.
    processing_time_ms : float
        Total wall-clock analysis time in milliseconds.
    metadata : dict
        Additional information: faces detected, frames analyzed,
        forensic metrics, model versions, errors, etc.
    """

    verdict: str
    confidence: float
    media_type: str
    real_confidence: float = 0.5
    fake_confidence: float = 0.5
    scores: Dict[str, Optional[float]] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Automatically sync real_confidence and fake_confidence if default
        if self.real_confidence == 0.5 and self.fake_confidence == 0.5:
            if self.verdict == "REAL":
                self.real_confidence = self.confidence
                self.fake_confidence = round(1.0 - self.confidence, 4)
            elif self.verdict == "FAKE":
                self.fake_confidence = self.confidence
                self.real_confidence = round(1.0 - self.confidence, 4)
            else:
                self.fake_confidence = self.confidence
                self.real_confidence = round(1.0 - self.confidence, 4)

    # ── Serialisation ─────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Convert the report to a plain dictionary."""
        return {
            "verdict": self.verdict,
            "confidence": round(self.confidence, 4),
            "real_confidence": round(self.real_confidence, 4),
            "fake_confidence": round(self.fake_confidence, 4),
            "media_type": self.media_type,
            "scores": {k: round(v, 4) if v is not None else None
                       for k, v in self.scores.items()},
            "processing_time_ms": round(self.processing_time_ms, 1),
            "metadata": self.metadata,
        }

    # ── Display helpers ───────────────────────

    @property
    def summary(self) -> str:
        """One-line human-readable summary."""
        scores_str = ", ".join(
            f"{k}={v:.4f}" for k, v in self.scores.items() if v is not None
        )
        return (
            f"Verdict: {self.verdict} (Real={self.real_confidence*100:.1f}%, Fake={self.fake_confidence*100:.1f}%) | "
            f"Media: {self.media_type} | Scores: [{scores_str}] | "
            f"Time: {self.processing_time_ms:.1f}ms"
        )

    @property
    def is_fake(self) -> bool:
        """Convenience flag."""
        return self.verdict == "FAKE"

    @property
    def is_real(self) -> bool:
        """Convenience flag."""
        return self.verdict == "REAL"
