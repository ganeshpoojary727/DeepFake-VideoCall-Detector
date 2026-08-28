"""
Audio deepfake analyzer — wraps the trained AASIST model for file-based detection.

Supports: .wav, .mp3, .flac, .ogg, .m4a, .aac, .wma
Provides rich diagnostic telemetry including temporal timeline and spectral forensics.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from app.analyzer.analysis_report import AnalysisReport
from app.audio.inference.voice_detector import VoiceDetector
from app.audio.preprocessing.audio_loader import AudioLoader
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Decision thresholds
_THRESHOLD_FAKE = 0.60
_THRESHOLD_REAL = 0.40


class AudioAnalyzer:
    """Audio deepfake analyzer using the AASIST architecture.

    Wraps the trained AASIST model for offline file-based audio deepfake detection,
    producing calibrated probabilities, timeline telemetry, and spectral cues.
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or settings.DEVICE
        self.target_sr: int = settings.audio.sample_rate  # 16000
        self.loader = AudioLoader(target_sr=self.target_sr)

        model_path = settings.project_root / "trained_models" / "audio" / "best_model.pt"
        logger.info("AudioAnalyzer: Initializing VoiceDetector with model path '%s'", model_path)
        self._detector = VoiceDetector(model_path=str(model_path), device=self.device)

    # ── Public Analysis APIs ───────────────────────────────────────────────────

    def analyze_structured(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Perform offline analysis on an uploaded file and return the exact structured schema.

        Returns
        -------
        Dict[str, Any]
            {
                "verdict": "REAL" | "FAKE",
                "confidence": float,
                "raw_scores": {
                    "bonafide_prob": float,
                    "spoof_prob": float
                },
                "spectral_cues": {
                    "peak_artifact_ranges": list,
                    "spectral_rolloff_hz": float,
                    "high_freq_energy_ratio": float,
                    "spectral_flatness": float,
                    "artifacts_detected": list
                },
                "timeline": list[dict]
            }
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        return self._detector.predict_file(path)

    def analyze(self, file_path: Union[str, Path]) -> AnalysisReport:
        """Analyze an audio file for deepfake content and return a unified AnalysisReport."""
        start = time.perf_counter()
        path = Path(file_path)

        try:
            structured_res = self.analyze_structured(path)

            verdict = structured_res["verdict"]
            confidence = structured_res["confidence"]
            raw_scores = structured_res["raw_scores"]
            spoof_prob = raw_scores["spoof_prob"]
            real_prob = raw_scores["bonafide_prob"]
            spectral_cues = structured_res["spectral_cues"]
            timeline = structured_res["timeline"]

            elapsed = (time.perf_counter() - start) * 1000.0

            logger.info(
                "AudioAnalyzer: %s → %s (Real=%.2f%%, Fake=%.2f%%, %.1fms)",
                path.name, verdict, real_prob * 100, spoof_prob * 100, elapsed,
            )

            # Metadata enriched with structured forensic fields
            metadata: Dict[str, Any] = {
                "file_name": path.name,
                "sample_rate": self.target_sr,
                "model": "AASIST (Graph Attention Network)",
                "raw_scores": raw_scores,
                "spectral_cues": spectral_cues,
                "timeline": timeline,
                "num_chunks_analyzed": len(timeline),
            }

            return AnalysisReport(
                verdict=verdict,
                confidence=round(confidence, 4),
                media_type="audio",
                real_confidence=round(real_prob, 4),
                fake_confidence=round(spoof_prob, 4),
                scores={"audio": round(spoof_prob, 4)},
                processing_time_ms=round(elapsed, 1),
                metadata=metadata,
            )

        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000.0
            logger.exception("AudioAnalyzer: Failed on %s: %s", path, exc)
            return AnalysisReport(
                verdict="UNCERTAIN",
                confidence=0.5,
                media_type="audio",
                real_confidence=0.5,
                fake_confidence=0.5,
                scores={"audio": None},
                processing_time_ms=round(elapsed, 1),
                metadata={
                    "error": str(exc),
                    "file_name": path.name,
                    "raw_scores": {"bonafide_prob": 0.5, "spoof_prob": 0.5},
                    "spectral_cues": {"peak_artifact_ranges": [], "artifacts_detected": []},
                    "timeline": [],
                },
            )

    def analyze_buffer(self, audio: np.ndarray, sr: int = 16000) -> float:
        """Return spoof probability from a raw numpy audio buffer."""
        return float(self._detector.predict_buffer(audio))

    @property
    def is_ready(self) -> bool:
        """Whether the AASIST model has been successfully initialized/loaded."""
        return getattr(self._detector, "_model_loaded", False)
