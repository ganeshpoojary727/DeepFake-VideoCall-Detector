"""
Unified Media Analyzer — Single File & Batch Deepfake Detection Orchestrator.

Handles routing, lazy model loading, unimodal and multimodal inference,
dynamic fusion, temporal synchronization, and consolidated reporting.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from app.analyzer.analysis_report import AnalysisReport, ConsolidatedForensicReport
from app.analyzer.media_router import (
    AUDIO_EXTENSIONS,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    MediaRouter,
    MediaType,
)
from app.config.settings import settings
from app.fusion.inference.fusion_engine import MultimodalFusion
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MediaAnalyzer:
    """Unified deepfake analysis engine for static files and multi-stream media.

    The central orchestrator for analyzing any media file (image, video, audio)
    with multimodal score fusion and synchronized telemetry.
    """

    def __init__(self, device: str = "auto") -> None:
        self._device = device
        self._image_analyzer = None
        self._video_analyzer = None
        self._audio_analyzer = None
        self._fusion = MultimodalFusion()
        self._router = MediaRouter()

    def _get_device(self) -> torch.device:
        """Resolve the computing device to be used."""
        if self._device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self._device)

    # ── Lazy Analyzer Properties ──────────────

    @property
    def image_analyzer(self):
        """Lazy-load and return the image analyzer."""
        if self._image_analyzer is None:
            from app.analyzer.image_analyzer import ImageAnalyzer
            self._image_analyzer = ImageAnalyzer(device=self._get_device())
        return self._image_analyzer

    @property
    def video_analyzer(self):
        """Lazy-load and return the video analyzer."""
        if self._video_analyzer is None:
            from app.analyzer.video_analyzer import VideoAnalyzer
            self._video_analyzer = VideoAnalyzer(device=self._get_device())
        return self._video_analyzer

    @property
    def audio_analyzer(self):
        """Lazy-load and return the audio analyzer."""
        if self._audio_analyzer is None:
            from app.analyzer.audio_analyzer import AudioAnalyzer
            self._audio_analyzer = AudioAnalyzer(device=self._get_device())
        return self._audio_analyzer

    # ── Consolidated & Unified Analysis APIs ──

    def analyze_consolidated(self, file_path: Union[str, Path]) -> ConsolidatedForensicReport:
        """Analyze any media file and return the standardized ConsolidatedForensicReport."""
        start = time.perf_counter()
        path = self._router.validate_file(file_path)
        media_type = self._router.detect_type(path)

        logger.info("MediaAnalyzer: Consolidated analysis for %s (%s)", path.name, media_type.value)

        # Generate natural language narrative synthesis
        from app.analyzer.forensic_explainer import GenerativeForensicExplainer
        explainer = GenerativeForensicExplainer()

        if media_type == MediaType.AUDIO:
            audio_telemetry = self.audio_analyzer.analyze_structured(path)
            elapsed = (time.perf_counter() - start) * 1000.0
            report = self._fusion.fuse_multimodal(
                audio_telemetry=audio_telemetry,
                visual_telemetry=None,
                media_type="AUDIO",
                processing_time_ms=elapsed,
                file_name=path.name,
            )

        elif media_type == MediaType.IMAGE:
            visual_telemetry = self.image_analyzer.analyze_structured(path)
            elapsed = (time.perf_counter() - start) * 1000.0
            report = self._fusion.fuse_multimodal(
                audio_telemetry=None,
                visual_telemetry=visual_telemetry,
                media_type="IMAGE",
                processing_time_ms=elapsed,
                file_name=path.name,
            )

        elif media_type == MediaType.VIDEO:
            visual_telemetry = self.video_analyzer.analyze_structured(path)

            # Try extracting and analyzing audio track from video
            audio_telemetry = None
            try:
                import librosa
                audio_arr, sr = librosa.load(str(path), sr=16000, mono=True)
                if audio_arr is not None and len(audio_arr) > 0:
                    audio_telemetry = self.audio_analyzer._detector.predict_detailed(audio_arr, sr=sr)
            except Exception as exc:
                logger.debug("MediaAnalyzer: Video audio demuxing skipped for %s: %s", path.name, exc)

            elapsed = (time.perf_counter() - start) * 1000.0
            media_str = "MULTIMODAL" if audio_telemetry is not None else "VIDEO"

            report = self._fusion.fuse_multimodal(
                audio_telemetry=audio_telemetry,
                visual_telemetry=visual_telemetry,
                media_type=media_str,
                processing_time_ms=elapsed,
                file_name=path.name,
            )

        else:
            raise ValueError(f"Unsupported media type: {media_type}")

        nl_report = explainer.explain(report)
        report.natural_language_report = nl_report.to_dict()
        return report

    def analyze(self, file_path: Union[str, Path]) -> AnalysisReport:
        """Analyze any single media file and return an enriched AnalysisReport."""
        path = self._router.validate_file(file_path)
        media_type = self._router.detect_type(path)

        if media_type == MediaType.IMAGE:
            return self.image_analyzer.analyze(path)
        elif media_type == MediaType.VIDEO:
            return self.video_analyzer.analyze(path)
        elif media_type == MediaType.AUDIO:
            return self.audio_analyzer.analyze(path)
        else:
            raise ValueError(f"Unsupported media type: {media_type}")

    def analyze_image(self, file_path: Union[str, Path]) -> AnalysisReport:
        return self.image_analyzer.analyze(file_path)

    def analyze_video(self, file_path: Union[str, Path]) -> AnalysisReport:
        return self.video_analyzer.analyze(file_path)

    def analyze_audio(self, file_path: Union[str, Path]) -> AnalysisReport:
        return self.audio_analyzer.analyze(file_path)

    # ── Batch & Directory Analysis ────────────

    def analyze_batch(
        self,
        file_paths: List[Union[str, Path]],
    ) -> List[AnalysisReport]:
        """Analyze a list of media files sequentially."""
        reports: List[AnalysisReport] = []
        for fp in file_paths:
            try:
                reports.append(self.analyze(fp))
            except Exception as exc:
                logger.error("Failed to analyze %s in batch: %s", fp, exc)
                reports.append(
                    AnalysisReport(
                        verdict="UNCERTAIN",
                        confidence=0.5,
                        media_type="unknown",
                        scores={},
                        processing_time_ms=0.0,
                        metadata={"error": str(exc), "file_name": str(fp)},
                    )
                )
        return reports

    def analyze_directory(
        self,
        dir_path: Union[str, Path],
        recursive: bool = False,
    ) -> List[AnalysisReport]:
        """Scan a directory for all supported media files and analyze them."""
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        all_extensions = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS
        pattern = "**/*" if recursive else "*"

        files_to_analyze = [
            p for p in dir_path.glob(pattern)
            if p.is_file() and p.suffix.lower() in all_extensions
        ]

        logger.info(
            "Directory scan found %d media files in %s",
            len(files_to_analyze), dir_path,
        )
        return self.analyze_batch(files_to_analyze)

    # ── System Health & Telemetry ─────────────

    def get_system_status(self) -> Dict[str, Any]:
        """Return system device, GPU memory, and model readiness status."""
        cuda_available = torch.cuda.is_available()
        status: Dict[str, Any] = {
            "device": str(self._get_device()),
            "cuda_available": cuda_available,
            "torch_version": torch.__version__,
        }

        if cuda_available:
            status["gpu_name"] = torch.cuda.get_device_name(0)
            status["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
            status["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024**2, 1)

        audio_model_path = settings.project_root / "trained_models" / "audio" / "best_model.pt"
        video_model_path = settings.project_root / "trained_models" / "video" / "best_model.pt"

        status["models"] = {
            "audio_aasist": {
                "checkpoint_exists": audio_model_path.exists(),
                "path": str(audio_model_path),
            },
            "video_efficientnet_transformer": {
                "checkpoint_exists": video_model_path.exists(),
                "path": str(video_model_path),
            },
        }

        return status
