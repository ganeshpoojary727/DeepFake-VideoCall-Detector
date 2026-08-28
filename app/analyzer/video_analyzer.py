"""
Video deepfake analyzer — spatiotemporal model + audio extraction + fusion.

Pipeline
--------
1. Decode video → sample 16 uniform frames → face detect + crop → EfficientNet-B4
   + Temporal Transformer → video fake probability.
2. Extract audio track via librosa → AASIST → audio fake probability.
3. Late fusion (0.6 × audio + 0.4 × video) → final verdict.

If no audio track is available, falls back to video-only analysis.

Supports: .mp4, .avi, .mkv, .mov, .webm, .wmv, .flv
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from app.analyzer.analysis_report import AnalysisReport
from app.config.settings import settings
from app.fusion.inference.fusion_engine import MultimodalFusion
from app.utils.logger import get_logger
from app.video.configs.inference_config import VideoInferenceConfig
from app.video.models.efficientnet.model import EfficientNetB4Model
from app.video.pipeline.inference_pipeline import InferencePipeline

logger = get_logger(__name__)

_THRESHOLD_FAKE = 0.65
_THRESHOLD_REAL = 0.35


class VideoAnalyzer:
    """Video deepfake analyzer with automatic audio track extraction
    and multimodal fusion.
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self._device = device or settings.DEVICE
        self._video_pipeline: Optional[InferencePipeline] = None
        self._audio_analyzer = None          # lazy-loaded
        self._fusion = MultimodalFusion()    # 0.6 audio + 0.4 video

    # ── Lazy initialisation ───────────────────

    def _ensure_pipeline_loaded(self) -> None:
        if self._video_pipeline is not None:
            return

        logger.info("VideoAnalyzer: loading EfficientNet-B4 + Temporal Transformer")
        config = VideoInferenceConfig()
        config.device = str(self._device)

        model = EfficientNetB4Model()
        weights_path = settings.project_root / "trained_models" / "video" / "best_model.pt"

        if weights_path.exists():
            model.load_weights(str(weights_path), strict=False)
            logger.info("VideoAnalyzer: loaded weights from %s", weights_path)
        else:
            logger.warning("VideoAnalyzer: weights not found at %s", weights_path)

        model.set_mode("inference")
        self._video_pipeline = InferencePipeline(model, config=config)
        logger.info("VideoAnalyzer: pipeline ready on %s", self._device)

    def _get_audio_analyzer(self):
        """Lazy-import AudioAnalyzer to avoid circular dependencies."""
        if self._audio_analyzer is None:
            from app.analyzer.audio_analyzer import AudioAnalyzer
            self._audio_analyzer = AudioAnalyzer(device=self._device)
        return self._audio_analyzer

    # ── Public API ────────────────────────────

    def analyze(self, file_path: Union[str, Path]) -> AnalysisReport:
        """Analyze a video file for deepfake content."""
        start = time.perf_counter()
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Video file not found: {file_path}")

        self._ensure_pipeline_loaded()

        # ── 1. Analyse video frames ───────────
        logger.info("VideoAnalyzer: analysing frames for %s", file_path.name)
        video_result = self._video_pipeline.predict_video(str(file_path))
        video_score = float(video_result.fake_probability)

        # ── 2. Try to extract and analyse audio ───
        audio_score = self._extract_and_analyze_audio(file_path)

        # ── 3. Fuse or fall back ──────────────
        if audio_score is not None:
            fusion = self._fusion.evaluate(audio_score, video_score)
            final_fake_prob = float(fusion["combined_score"])
            method = "multimodal_fusion"
        else:
            final_fake_prob = video_score
            method = "video_only"

        final_fake_prob = float(np.clip(final_fake_prob, 0.01, 0.99))
        final_real_prob = float(round(1.0 - final_fake_prob, 4))

        # ── 4. Three-way verdict ──────────────
        if final_fake_prob >= _THRESHOLD_FAKE:
            verdict = "FAKE"
            verdict_confidence = final_fake_prob
        elif final_real_prob >= (1.0 - _THRESHOLD_REAL):
            verdict = "REAL"
            verdict_confidence = final_real_prob
        else:
            verdict = "UNCERTAIN"
            verdict_confidence = max(final_real_prob, final_fake_prob)

        elapsed = (time.perf_counter() - start) * 1000.0

        logger.info(
            "VideoAnalyzer: %s → %s (Real=%.2f%%, Fake=%.2f%%, video=%.4f, audio=%s, %.1fms)",
            file_path.name, verdict, final_real_prob * 100, final_fake_prob * 100, video_score,
            f"{audio_score:.4f}" if audio_score is not None else "N/A",
            elapsed,
        )

        return AnalysisReport(
            verdict=verdict,
            confidence=round(verdict_confidence, 4),
            media_type="video",
            real_confidence=round(final_real_prob, 4),
            fake_confidence=round(final_fake_prob, 4),
            scores={
                "video": round(video_score, 4),
                "audio": round(audio_score, 4) if audio_score is not None else None,
                "fused": round(final_fake_prob, 4) if audio_score is not None else None,
            },
            processing_time_ms=round(elapsed, 1),
            metadata={
                "file_name": file_path.name,
                "analysis_method": method,
                "num_frames": getattr(video_result, "num_frames", None),
                "num_faces_detected": getattr(video_result, "num_faces_detected", None),
                "video_inference_ms": getattr(video_result, "total_runtime_ms", None),
                "model": "EfficientNet-B4 + Temporal Transformer",
            },
        )

    # ── Private helpers ───────────────────────

    def _extract_and_analyze_audio(self, video_path: Path) -> Optional[float]:
        """Extract audio track from a video file and return spoof probability."""
        try:
            import librosa
            audio, sr = librosa.load(str(video_path), sr=16000, mono=True)

            if audio is None or len(audio) == 0:
                logger.info("VideoAnalyzer: no audio track in %s", video_path.name)
                return None

            analyzer = self._get_audio_analyzer()
            return analyzer.analyze_buffer(audio, sr=16000)

        except Exception as exc:
            logger.debug("VideoAnalyzer: audio extraction failed for %s: %s", video_path.name, exc)
            return None
