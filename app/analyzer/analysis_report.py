"""
Unified analysis report and Consolidated Forensic Report for multimodal deepfake detection.

Provides:
1. ConsolidatedForensicReport: Standardized Phase 3 multimodal diagnostic report schema.
2. AnalysisReport: Unified per-analyzer report with bidirectional conversion helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class ConsolidatedForensicReport:
    """Consolidated Multimodal Forensic Report schema for Phase 3 offline file analysis.

    Attributes
    ----------
    media_type : str
        "AUDIO" | "IMAGE" | "VIDEO" | "MULTIMODAL"
    verdict : str
        "REAL" | "FAKE"
    overall_confidence : float
        Calibrated decision confidence in [0.0, 1.0].
    modality_breakdown : dict
        {
            "audio": Optional[Dict[str, Any]],
            "visual": Optional[Dict[str, Any]],
            "classical_forensics": Optional[Dict[str, Any]]
        }
    temporal_sync : list of dict
        Aligned second-by-second timestamps mapping audio vs. visual spoof probabilities.
    top_anomalies : list of dict
        Prioritized list of time segments / frames with highest manipulation evidence.
    processing_time_ms : float
        End-to-end inference and fusion latency.
    metadata : dict
        Additional metadata (file name, duration, models used, etc.).
    """

    media_type: str
    verdict: str
    overall_confidence: float
    modality_breakdown: Dict[str, Optional[Dict[str, Any]]] = field(default_factory=dict)
    temporal_sync: List[Dict[str, Any]] = field(default_factory=list)
    top_anomalies: List[Dict[str, Any]] = field(default_factory=list)
    natural_language_report: Optional[Dict[str, Any]] = None
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to standardized JSON-serializable dictionary."""
        return {
            "media_type": self.media_type,
            "verdict": self.verdict,
            "overall_confidence": round(float(self.overall_confidence), 4),
            "modality_breakdown": self.modality_breakdown,
            "temporal_sync": self.temporal_sync,
            "top_anomalies": self.top_anomalies,
            "natural_language_report": self.natural_language_report,
            "processing_time_ms": round(float(self.processing_time_ms), 1),
            "metadata": self.metadata,
        }

    @property
    def is_fake(self) -> bool:
        return self.verdict == "FAKE"

    @property
    def is_real(self) -> bool:
        return self.verdict == "REAL"

    @property
    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"ConsolidatedForensicReport: Verdict={self.verdict} ({self.overall_confidence*100:.1f}%) | "
            f"Type={self.media_type} | Anomalies={len(self.top_anomalies)} | "
            f"Time={self.processing_time_ms:.1f}ms"
        )


@dataclass
class AnalysisReport:
    """Unified result returned by media analyzers with dual-sided confidence."""

    verdict: str                               # "REAL" | "FAKE" | "UNCERTAIN" | "NOT_APPLICABLE"
    confidence: float
    media_type: str
    real_confidence: float = 0.5
    fake_confidence: float = 0.5
    scores: Dict[str, Optional[float]] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    content_category: Optional[str] = None    # Set for NOT_APPLICABLE verdicts (e.g. "DIGITAL_ART_ANIME")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to plain dictionary."""
        return {
            "verdict": self.verdict,
            "confidence": round(self.confidence, 4),
            "real_confidence": round(self.real_confidence, 4),
            "fake_confidence": round(self.fake_confidence, 4),
            "media_type": self.media_type,
            "content_category": self.content_category,
            "scores": {k: round(v, 4) if v is not None else None for k, v in self.scores.items()},
            "processing_time_ms": round(self.processing_time_ms, 1),
            "metadata": self.metadata,
        }

    @property
    def summary(self) -> str:
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
        return self.verdict == "FAKE"

    @property
    def is_real(self) -> bool:
        return self.verdict == "REAL"

    @property
    def is_uncertain(self) -> bool:
        return self.verdict == "UNCERTAIN"

    @property
    def is_not_applicable(self) -> bool:
        """True when Stage-0 determined the media is non-biometric artwork."""
        return self.verdict == "NOT_APPLICABLE"
