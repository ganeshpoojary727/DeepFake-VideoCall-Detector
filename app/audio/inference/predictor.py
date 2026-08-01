"""
Reusable prediction interface for audio deepfake detection.

Improvements over v1
─────────────────────
• Returns ``PredictionResult`` dataclass (not a raw dict)
• Confidence as [0, 1] probability (not percentage)
• Configurable confidence threshold with ``UNCERTAIN`` label
• Input validation
• Latency tracking
• Consistent 4-space indentation
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.audio.preprocessing.audio_preprocessor import AudioPreprocessor
from app.audio.features.feature_extractor import FeatureExtractor
from app.config.settings import settings
from app.core.interfaces import DetectionLabel, Modality, PredictionResult
from app.utils.helpers import validate_audio_file
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    """
    Single-file audio deepfake prediction.

    Parameters
    ----------
    model : nn.Module
        Trained model (must be on ``device``).
    device : torch.device
        Compute device.
    threshold_fake : float
        Spoof-class probability above which the sample is labelled FAKE.
    threshold_real : float
        Spoof-class probability below which the sample is labelled REAL.
        Between the two thresholds → UNCERTAIN.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        threshold_fake: Optional[float] = None,
        threshold_real: Optional[float] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.threshold_fake = (
            threshold_fake or settings.inference.confidence_threshold_fake
        )
        self.threshold_real = (
            threshold_real or settings.inference.confidence_threshold_real
        )
        self.preprocessor = AudioPreprocessor()
        self.extractor = FeatureExtractor()

    def predict(self, audio_path: str | Path) -> PredictionResult:
        """
        Run inference on a single audio file.

        Parameters
        ----------
        audio_path : str | Path
            Path to the audio file.

        Returns
        -------
        PredictionResult
            Structured prediction with label, confidence, and latency.
        """
        # ── Validate input ────────────────────
        audio_path = validate_audio_file(audio_path)

        start = time.perf_counter()

        self.model.eval()
        with torch.no_grad():
            # Preprocess
            audio, _ = self.preprocessor.preprocess(audio_path)

            # Extract features
            feature = self.extractor.extract(audio)
            feature = feature.unsqueeze(0).to(self.device)  # add batch dim

            # Forward pass
            output = self.model(feature)

            # Convert logits to probabilities
            probabilities = F.softmax(output, dim=1)
            spoof_prob = probabilities[0, 1].item()  # P(spoof)

        latency = (time.perf_counter() - start) * 1000  # ms

        # ── Three-way decision ────────────────
        if spoof_prob >= self.threshold_fake:
            label = DetectionLabel.FAKE
        elif spoof_prob <= self.threshold_real:
            label = DetectionLabel.REAL
        else:
            label = DetectionLabel.UNCERTAIN

        result = PredictionResult(
            label=label,
            confidence=spoof_prob,
            modality=Modality.AUDIO,
            latency_ms=round(latency, 2),
            model_version=settings.model.model_version,
        )

        logger.debug(
            "Prediction: %s (conf=%.3f, latency=%.1fms)",
            result.label.value,
            result.confidence,
            result.latency_ms,
        )

        return result