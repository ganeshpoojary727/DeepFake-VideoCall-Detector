"""
Audio deepfake analyzer — wraps the trained AASIST model for file-based detection.

Supports: .wav, .mp3, .flac, .ogg, .m4a, .aac, .wma
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from app.analyzer.analysis_report import AnalysisReport
from app.audio.inference.voice_detector import VoiceDetector
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Thresholds for three-way decision
_THRESHOLD_FAKE = 0.70
_THRESHOLD_REAL = 0.30


class AudioAnalyzer:
    """Audio deepfake analyzer using the AASIST model.

    Wraps the trained AASIST model for file-based audio deepfake detection.
    Loads the best checkpoint from ``trained_models/audio/best_model.pt``.
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or settings.DEVICE
        self.target_sr: int = settings.audio.sample_rate  # 16000

        model_path = settings.project_root / "trained_models" / "audio" / "best_model.pt"
        logger.info("AudioAnalyzer: loading AASIST from %s", model_path)
        self._detector = VoiceDetector(model_path=str(model_path), device=self.device)

    # ── Public API ────────────────────────────

    def analyze(self, file_path: str | Path) -> AnalysisReport:
        """Analyze an audio file for deepfake content.

        Parameters
        ----------
        file_path : str | Path
            Path to the audio file.

        Returns
        -------
        AnalysisReport
        """
        start = time.perf_counter()
        file_path = Path(file_path)

        try:
            audio, sr = self._load_audio(file_path)

            if sr != self.target_sr:
                audio = self._resample(audio, sr, self.target_sr)

            spoof_prob = float(self._detector.predict_buffer(audio))

            if spoof_prob >= _THRESHOLD_FAKE:
                verdict = "FAKE"
            elif spoof_prob <= _THRESHOLD_REAL:
                verdict = "REAL"
            else:
                verdict = "UNCERTAIN"

            elapsed = (time.perf_counter() - start) * 1000.0

            logger.info(
                "AudioAnalyzer: %s → %s (spoof_prob=%.4f, %.1fms)",
                file_path.name, verdict, spoof_prob, elapsed,
            )

            return AnalysisReport(
                verdict=verdict,
                confidence=spoof_prob,
                media_type="audio",
                scores={"audio": spoof_prob},
                processing_time_ms=round(elapsed, 1),
                metadata={
                    "file_name": file_path.name,
                    "sample_rate": self.target_sr,
                    "duration_seconds": round(len(audio) / self.target_sr, 2),
                    "model": "AASIST",
                },
            )

        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000.0
            logger.exception("AudioAnalyzer: failed on %s: %s", file_path, exc)
            return AnalysisReport(
                verdict="UNCERTAIN",
                confidence=0.5,
                media_type="audio",
                scores={"audio": None},
                processing_time_ms=round(elapsed, 1),
                metadata={"error": str(exc), "file_name": file_path.name},
            )

    def analyze_buffer(self, audio: np.ndarray, sr: int = 16000) -> float:
        """Return spoof probability from a raw numpy buffer.

        Used internally by VideoAnalyzer for extracted audio tracks.

        Returns
        -------
        float
            Spoof probability in ``[0.0, 1.0]``.
        """
        if sr != self.target_sr:
            audio = self._resample(audio, sr, self.target_sr)
        return float(self._detector.predict_buffer(audio))

    # ── Private helpers ───────────────────────

    def _load_audio(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio with soundfile (primary) / librosa (fallback)."""
        try:
            import soundfile as sf
            audio, sr = sf.read(str(file_path), dtype="float32")
        except Exception:
            import librosa
            logger.debug("soundfile failed, falling back to librosa")
            audio, sr = librosa.load(str(file_path), sr=None, mono=True)
            return audio.astype(np.float32), int(sr)

        # Stereo → mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Peak-normalise to [-1, 1]
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak

        return audio.astype(np.float32), int(sr)

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # numpy-based fallback
            duration = len(audio) / orig_sr
            target_len = int(duration * target_sr)
            indices = np.linspace(0, len(audio) - 1, target_len)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    @property
    def is_ready(self) -> bool:
        """Whether the AASIST model has been loaded."""
        return getattr(self._detector, "_model_loaded", False)
