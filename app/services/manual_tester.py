"""
Manual Media Tester — analyze any audio/video file for deepfake detection.

Supports:
    Audio: .wav, .mp3, .flac, .ogg, .m4a
    Video: .mp4, .avi, .mkv, .mov, .webm (extracts audio track + video frames)

Returns a detailed report dictionary with audio, video, and combined scores.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.fusion.inference.fusion_engine import MultimodalFusion
from app.audio.inference.voice_detector import VoiceDetector
from app.video.inference.video_detector import VideoDeepfakeDetector
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# File extensions
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".wmv", ".flv"}


class ManualMediaTester:
    """
    Analyze any audio or video file for deepfake detection.

    Extracts audio and video tracks, runs both detectors,
    and computes the multimodal fusion score.
    """

    def __init__(self) -> None:
        self._voice_detector = VoiceDetector()
        self._video_detector = VideoDeepfakeDetector()
        self._fusion = MultimodalFusion()

    def analyze_file(self, file_path: str | Path) -> Dict[str, object]:
        """
        Analyze a media file for deepfake content.

        Parameters
        ----------
        file_path : str or Path
            Path to the audio or video file.

        Returns
        -------
        dict
            Detailed report with keys:
            - ``file_path``: str
            - ``file_type``: "audio" or "video"
            - ``audio_score``: float (0-1) or None if no audio
            - ``video_score``: float (0-1) or None if no video
            - ``combined_score``: float (0-1)
            - ``prediction``: "REAL" or "DEEPFAKE"
            - ``analysis_time_ms``: float
            - ``details``: dict with additional info
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}

        start = time.perf_counter()
        ext = file_path.suffix.lower()

        audio_score = None
        video_score = None
        file_type = "unknown"

        if ext in AUDIO_EXTENSIONS:
            file_type = "audio"
            audio_score = self._analyze_audio(file_path)

        elif ext in VIDEO_EXTENSIONS:
            file_type = "video"
            audio_score = self._extract_and_analyze_audio(file_path)
            video_score = self._extract_and_analyze_video(file_path)

        else:
            # Try as video first, then audio
            video_score = self._extract_and_analyze_video(file_path)
            if video_score is not None:
                file_type = "video"
                audio_score = self._extract_and_analyze_audio(file_path)
            else:
                file_type = "audio"
                audio_score = self._analyze_audio(file_path)

        # Default scores for missing modalities
        if audio_score is None:
            audio_score = 0.5
        if video_score is None:
            video_score = 0.5

        # Compute fusion
        fusion_result = self._fusion.evaluate(audio_score, video_score)

        elapsed = (time.perf_counter() - start) * 1000

        report = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_type": file_type,
            "audio_score": round(audio_score, 4),
            "video_score": round(video_score, 4),
            "combined_score": fusion_result["combined_score"],
            "prediction": fusion_result["prediction"],
            "analysis_time_ms": round(elapsed, 1),
            "details": {
                "audio_weight": fusion_result["audio_weight"],
                "video_weight": fusion_result["video_weight"],
            },
        }

        logger.info(
            "Manual analysis: %s → %s (combined=%.4f, audio=%.4f, video=%.4f, %.1fms)",
            file_path.name,
            report["prediction"],
            report["combined_score"],
            audio_score,
            video_score,
            elapsed,
        )

        return report

    def _analyze_audio(self, file_path: Path) -> Optional[float]:
        """Load and analyze an audio file."""
        try:
            import soundfile as sf

            audio_data, sr = sf.read(str(file_path), dtype="float32")
            if audio_data.ndim > 1:
                audio_data = audio_data[:, 0]  # Take first channel

            # Resample if needed
            if sr != settings.audio.sample_rate:
                audio_data = self._resample(audio_data, sr, settings.audio.sample_rate)

            return self._voice_detector.predict_from_buffer(
                audio_data, sr=settings.audio.sample_rate
            )
        except Exception as exc:
            logger.debug("Audio analysis failed for %s: %s", file_path, exc)
            return None

    def _extract_and_analyze_audio(self, video_path: Path) -> Optional[float]:
        """Extract audio track from a video file and analyze it."""
        try:
            import soundfile as sf

            # Try reading audio directly (works for some containers)
            try:
                audio_data, sr = sf.read(str(video_path), dtype="float32")
                if audio_data.ndim > 1:
                    audio_data = audio_data[:, 0]
                if sr != settings.audio.sample_rate:
                    audio_data = self._resample(audio_data, sr, settings.audio.sample_rate)
                return self._voice_detector.predict_from_buffer(
                    audio_data, sr=settings.audio.sample_rate
                )
            except Exception:
                pass

            # Fallback: use cv2 to read video, no audio extraction possible
            # Try librosa as alternative
            try:
                import librosa
                audio_data, sr = librosa.load(
                    str(video_path),
                    sr=settings.audio.sample_rate,
                    mono=True,
                )
                return self._voice_detector.predict_from_buffer(
                    audio_data, sr=settings.audio.sample_rate
                )
            except Exception:
                pass

            return None
        except Exception as exc:
            logger.debug("Audio extraction failed for %s: %s", video_path, exc)
            return None

    def _extract_and_analyze_video(self, video_path: Path) -> Optional[float]:
        """Extract frames from a video file and analyze them."""
        try:
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None

            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Sample ~150 frames (30s at 5fps equivalent)
            target_frames = min(150, total_frames)
            step = max(1, total_frames // target_frames)

            frames: List[np.ndarray] = []
            frame_idx = 0

            while cap.isOpened() and len(frames) < target_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % step == 0:
                    frames.append(frame)
                frame_idx += 1

            cap.release()

            if len(frames) < 3:
                return None

            return self._video_detector.predict_from_frames(frames)

        except Exception as exc:
            logger.debug("Video analysis failed for %s: %s", video_path, exc)
            return None

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling using numpy interpolation."""
        if orig_sr == target_sr:
            return audio
        duration = len(audio) / orig_sr
        target_len = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
