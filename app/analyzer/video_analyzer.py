"""
Video Deepfake Analyzer — Spatiotemporal Sequence Model + Multimodal Audio Fusion + Forensics.

Pipeline:
1. Decode video → sample 16/32 uniform frames → face detect + crop → EfficientNet-B4 + Temporal Attention.
2. Classical Forensics (ELA, 2D FFT azimuthal spectrum, boundary Laplacian inconsistency).
3. Grad-CAM visual explainability on top anomalous keyframes.
4. Extract audio track via librosa → AASIST → audio fake probability.
5. Multimodal late fusion (0.6 × audio + 0.4 × video) → final verdict.

Supports: .mp4, .avi, .mkv, .mov, .webm, .wmv, .flv
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from app.analyzer.analysis_report import AnalysisReport
from app.config.settings import settings
from app.fusion.inference.fusion_engine import MultimodalFusion
from app.utils.logger import get_logger
from app.video.inference.video_detector import VideoDetector

logger = get_logger(__name__)

_THRESHOLD_FAKE = 0.60
_THRESHOLD_REAL = 0.40


class VideoAnalyzer:
    """Video deepfake analyzer with multi-signal telemetry, Grad-CAM explainability,
    and automatic audio track extraction.
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self._device = device or settings.DEVICE
        self._video_detector: Optional[VideoDetector] = None
        self._audio_analyzer = None
        self._fusion = MultimodalFusion()

    # ── Lazy initialisation ───────────────────

    def _ensure_detector_loaded(self) -> None:
        if self._video_detector is not None:
            return

        logger.info("VideoAnalyzer: Initializing VideoDetector with EfficientNet-B4")
        weights_path = settings.project_root / "trained_models" / "video" / "best_model.pt"
        self._video_detector = VideoDetector(
            model_path=weights_path if weights_path.exists() else None,
            device=self._device,
            sequence_length=16,
        )

    def _get_audio_analyzer(self):
        """Lazy-import AudioAnalyzer to avoid circular dependencies."""
        if self._audio_analyzer is None:
            from app.analyzer.audio_analyzer import AudioAnalyzer
            self._audio_analyzer = AudioAnalyzer(device=self._device)
        return self._audio_analyzer

    # ── Public APIs ───────────────────────────

    def analyze_structured(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Perform offline analysis on an uploaded video and return standardized telemetry schema.

        Returns
        -------
        Dict[str, Any]
            Standardized telemetry dictionary:
            - verdict: "REAL" | "FAKE"
            - confidence: float
            - raw_scores: {"real_prob": float, "fake_prob": float}
            - visual_cues: {"ela_discrepancy_score": float, "fft_spectral_anomaly": float, "boundary_inconsistency": float}
            - timeline: list of per-frame predictions with timestamps
            - key_artifacts: list of top anomalous frames with bbox and Grad-CAM saliency
        """
        self._ensure_detector_loaded()
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        return self._video_detector.predict_video(path)

    def analyze(self, file_path: Union[str, Path]) -> AnalysisReport:
        """Analyze a video file for deepfake content and return a unified AnalysisReport."""
        start = time.perf_counter()
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        self._ensure_detector_loaded()

        # 1. Video frame & forensics analysis
        logger.info("VideoAnalyzer: Analyzing video frames for %s", path.name)
        structured_video = self.analyze_structured(path)
        video_fake_score = structured_video["raw_scores"]["fake_prob"]

        # 2. Audio track extraction & analysis
        audio_score = self._extract_and_analyze_audio(path)

        # 3. Multimodal fusion
        if audio_score is not None:
            fusion = self._fusion.evaluate(audio_score, video_fake_score)
            final_fake_prob = float(fusion["combined_score"])
            method = "multimodal_fusion"
        else:
            final_fake_prob = video_fake_score
            method = "video_only"

        final_fake_prob = float(np.clip(final_fake_prob, 0.01, 0.99))
        final_real_prob = float(round(1.0 - final_fake_prob, 4))

        # 4. Verdict determination
        if final_fake_prob >= _THRESHOLD_FAKE:
            verdict = "FAKE"
            confidence = final_fake_prob
        elif final_real_prob >= (1.0 - _THRESHOLD_REAL):
            verdict = "REAL"
            confidence = final_real_prob
        else:
            verdict = "UNCERTAIN"
            confidence = max(final_real_prob, final_fake_prob)

        elapsed = (time.perf_counter() - start) * 1000.0

        logger.info(
            "VideoAnalyzer: %s → %s (Real=%.2f%%, Fake=%.2f%%, video=%.4f, audio=%s, %.1fms)",
            path.name, verdict, final_real_prob * 100, final_fake_prob * 100, video_fake_score,
            f"{audio_score:.4f}" if audio_score is not None else "N/A",
            elapsed,
        )

        metadata: Dict[str, Any] = {
            "file_name": path.name,
            "analysis_method": method,
            "raw_scores": {
                "real_prob": round(final_real_prob, 4),
                "fake_prob": round(final_fake_prob, 4),
            },
            "visual_cues": structured_video["visual_cues"],
            "timeline": structured_video["timeline"],
            "key_artifacts": structured_video["key_artifacts"],
            "num_frames_analyzed": len(structured_video["timeline"]),
            "model": "EfficientNet-B4 + Temporal Attention & Forensic Engine",
        }

        return AnalysisReport(
            verdict=verdict,
            confidence=round(confidence, 4),
            media_type="video",
            real_confidence=round(final_real_prob, 4),
            fake_confidence=round(final_fake_prob, 4),
            scores={
                "video": round(video_fake_score, 4),
                "audio": round(audio_score, 4) if audio_score is not None else None,
                "fused": round(final_fake_prob, 4) if audio_score is not None else None,
            },
            processing_time_ms=round(elapsed, 1),
            metadata=metadata,
        )

    # ── Private helpers ───────────────────────

    def _extract_and_analyze_audio(self, video_path: Path) -> Optional[float]:
        """Extract audio track from video file and return spoof probability."""
        try:
            import librosa
            audio, sr = librosa.load(str(video_path), sr=16000, mono=True)

            if audio is None or len(audio) == 0:
                logger.info("VideoAnalyzer: No audio track in %s", video_path.name)
                return None

            analyzer = self._get_audio_analyzer()
            return analyzer.analyze_buffer(audio, sr=16000)

        except Exception as exc:
            logger.debug("VideoAnalyzer: Audio extraction skipped for %s: %s", video_path.name, exc)
            return None
