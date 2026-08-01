"""
Core interfaces, data classes, and enums for the DeepFake Video Call Detector.

This module defines the foundational abstractions that all AI components
implement. It has NO external dependencies beyond the standard library,
ensuring a clean dependency graph (domain layer depends on nothing).
"""

from __future__ import annotations

import enum
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import numpy as np
import torch


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────


class DetectionLabel(enum.Enum):
    """Three-way classification result."""

    REAL = "REAL"
    FAKE = "FAKE"
    UNCERTAIN = "UNCERTAIN"


class Modality(enum.Enum):
    """Sensory channel used for detection."""

    AUDIO = "audio"
    VIDEO = "video"
    FUSED = "fused"


# ──────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────


@dataclass
class PredictionResult:
    """Immutable result returned by any detector."""

    label: DetectionLabel
    confidence: float  # 0.0 – 1.0
    modality: Modality
    latency_ms: float = 0.0
    model_version: str = "unknown"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be in [0, 1], got {self.confidence}"
            )


@dataclass
class EvaluationResult:
    """Structured evaluation metrics (replaces fragile tuple)."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    eer: Optional[float]  # Equal Error Rate — standard for anti-spoofing
    confusion_matrix: Any  # numpy ndarray
    classification_report: str


@dataclass
class AudioSegment:
    """A chunk of audio ready for processing."""

    waveform: np.ndarray
    sample_rate: int
    duration_seconds: float = field(init=False)

    def __post_init__(self) -> None:
        self.duration_seconds = len(self.waveform) / self.sample_rate


# ──────────────────────────────────────────────
# Abstract Base Classes
# ──────────────────────────────────────────────


class BasePreprocessor(ABC):
    """Interface for all audio / video preprocessors."""

    @abstractmethod
    def process(self, raw_input: Any) -> Any:
        """Process raw input into a format ready for feature extraction."""
        ...


class BaseFeatureExtractor(ABC):
    """Interface for all feature extractors."""

    @abstractmethod
    def extract(self, processed_input: Any) -> torch.Tensor:
        """Extract model-ready features from preprocessed input."""
        ...


class BaseDetector(ABC):
    """Interface for all detectors (audio, video, fused)."""

    @abstractmethod
    def detect(self, input_data: Any) -> PredictionResult:
        """Run detection on a single input."""
        ...

    @abstractmethod
    def detect_stream(self, stream: Any) -> Iterator[PredictionResult]:
        """Run continuous detection on a live stream."""
        ...
