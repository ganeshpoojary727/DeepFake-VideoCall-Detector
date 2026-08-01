"""
Prediction manager — coordinates audio and video detectors with late fusion.

This module manages the lifecycle of detection models and provides
a unified interface for single-modality or fused predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.audio.inference.predictor import Predictor
from app.video.inference.video_detector import VideoDetector
from app.config.settings import settings
from app.core.interfaces import DetectionLabel, Modality, PredictionResult
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionManager:
    """
    High-level prediction coordinator.

    Manages audio and (future) video detectors, providing a single
    ``predict()`` entry point that handles fusion when both modalities
    are available.

    Parameters
    ----------
    audio_predictor : Predictor | None
        Audio deepfake predictor.
    video_detector : VideoDetector | None
        Video deepfake detector (currently a skeleton).
    """

    def __init__(
        self,
        audio_predictor: Optional[Predictor] = None,
        video_detector: Optional[VideoDetector] = None,
    ) -> None:
        self.audio_predictor = audio_predictor
        self.video_detector = video_detector

    @property
    def has_audio(self) -> bool:
        """Whether an audio predictor is available."""
        return self.audio_predictor is not None

    @property
    def has_video(self) -> bool:
        """Whether a video detector is available."""
        return self.video_detector is not None

    def predict_audio(self, audio_path: str | Path) -> PredictionResult:
        """
        Run audio-only prediction.

        Raises
        ------
        RuntimeError
            If no audio predictor is configured.
        """
        if not self.has_audio:
            raise RuntimeError("No audio predictor configured.")
        return self.audio_predictor.predict(audio_path)

    def predict_fused(
        self,
        audio_path: Optional[str | Path] = None,
        video_input: Optional[object] = None,
    ) -> PredictionResult:
        """
        Run late-fusion prediction combining audio and video results.

        If only one modality is available, its result is returned directly.

        Parameters
        ----------
        audio_path : str | Path | None
            Audio file for audio prediction.
        video_input : object | None
            Video input for video prediction (future).

        Returns
        -------
        PredictionResult
            Fused or single-modality prediction.

        Raises
        ------
        ValueError
            If no input is provided for any modality.
        """
        audio_result: Optional[PredictionResult] = None
        video_result: Optional[PredictionResult] = None

        # Audio prediction
        if audio_path is not None and self.has_audio:
            try:
                audio_result = self.audio_predictor.predict(audio_path)
            except Exception as exc:
                logger.warning("Audio prediction failed: %s", exc)

        # Video prediction (future)
        if video_input is not None and self.has_video:
            try:
                video_result = self.video_detector.detect(video_input)
            except NotImplementedError:
                logger.debug("Video detection not yet implemented")
            except Exception as exc:
                logger.warning("Video prediction failed: %s", exc)

        # ── Fusion logic ──────────────────────
        if audio_result and video_result:
            return self._fuse(audio_result, video_result)
        elif audio_result:
            return audio_result
        elif video_result:
            return video_result
        else:
            raise ValueError("No valid predictions from any modality.")

    def _fuse(
        self,
        audio: PredictionResult,
        video: PredictionResult,
    ) -> PredictionResult:
        """
        Late fusion via weighted average of confidence scores.

        Uses ``settings.inference.audio_fusion_weight`` and
        ``settings.inference.video_fusion_weight``.
        """
        w_a = settings.inference.audio_fusion_weight
        w_v = settings.inference.video_fusion_weight
        fused_score = w_a * audio.confidence + w_v * video.confidence

        if fused_score >= settings.inference.confidence_threshold_fake:
            label = DetectionLabel.FAKE
        elif fused_score <= settings.inference.confidence_threshold_real:
            label = DetectionLabel.REAL
        else:
            label = DetectionLabel.UNCERTAIN

        latency = audio.latency_ms + video.latency_ms

        logger.info(
            "Fusion: audio=%.3f × %.1f + video=%.3f × %.1f = %.3f → %s",
            audio.confidence, w_a, video.confidence, w_v, fused_score, label.value,
        )

        return PredictionResult(
            label=label,
            confidence=fused_score,
            modality=Modality.FUSED,
            latency_ms=round(latency, 2),
            model_version=audio.model_version,
        )
