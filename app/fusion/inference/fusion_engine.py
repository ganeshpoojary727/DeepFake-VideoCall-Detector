"""
Multimodal Fusion Engine — Uncertainty-Weighted Score Fusion & Cross-Modal Telemetry Synchronization.

Unifies Audio AASIST, Visual EfficientNet, and Classical Forensics (ELA, 2D FFT, Laplacian)
into a consolidated, second-by-second aligned forensic diagnosis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from app.analyzer.analysis_report import ConsolidatedForensicReport
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MultimodalFusion:
    """Multimodal Deepfake Fusion Engine with Dynamic Gating & Temporal Synchronization."""

    def __init__(
        self,
        audio_weight: Optional[float] = None,
        video_weight: Optional[float] = None,
        threshold: float = 0.50,
        anomaly_threshold: float = 0.70,
    ) -> None:
        self.audio_weight = audio_weight if audio_weight is not None else settings.AUDIO_WEIGHT
        self.video_weight = video_weight if video_weight is not None else settings.VIDEO_WEIGHT
        self.threshold = threshold
        self.anomaly_threshold = anomaly_threshold

    # ── Backward-compatible Simple Score Fusion ────────────────────────────────

    def evaluate(self, audio_score: float, video_score: float) -> Dict[str, Any]:
        """Compute weighted fusion of audio and video scores (backward-compatible)."""
        combined = (self.audio_weight * audio_score) + (self.video_weight * video_score)
        combined = float(np.clip(combined, 0.0, 1.0))
        prediction = "DEEPFAKE" if combined >= self.threshold else "REAL"

        return {
            "combined_score": round(combined, 4),
            "prediction": prediction,
            "audio_score": round(float(audio_score), 4),
            "video_score": round(float(video_score), 4),
            "audio_weight": self.audio_weight,
            "video_weight": self.video_weight,
        }

    # ── Phase 3 Consolidated Multimodal Fusion ─────────────────────────────────

    def fuse_multimodal(
        self,
        audio_telemetry: Optional[Dict[str, Any]] = None,
        visual_telemetry: Optional[Dict[str, Any]] = None,
        media_type: str = "MULTIMODAL",
        processing_time_ms: float = 0.0,
        file_name: str = "",
    ) -> ConsolidatedForensicReport:
        """Fuse unimodal or multimodal telemetry streams into a ConsolidatedForensicReport.

        Parameters
        ----------
        audio_telemetry : Dict[str, Any] | None
            Output from AudioAnalyzer / VoiceDetector.
        visual_telemetry : Dict[str, Any] | None
            Output from VideoAnalyzer / ImageAnalyzer / VideoDetector.
        media_type : str
            "AUDIO", "IMAGE", "VIDEO", or "MULTIMODAL".
        processing_time_ms : float
            End-to-end processing latency.
        file_name : str
            Source media filename.

        Returns
        -------
        ConsolidatedForensicReport
            Standardized consolidated diagnostic report.
        """
        # Case 1: Audio Only
        if audio_telemetry is not None and visual_telemetry is None:
            return self._fuse_audio_only(audio_telemetry, processing_time_ms, file_name)

        # Case 2: Visual Only (Image or Video without audio)
        if visual_telemetry is not None and audio_telemetry is None:
            actual_type = "IMAGE" if media_type.upper() == "IMAGE" else "VIDEO"
            return self._fuse_visual_only(visual_telemetry, actual_type, processing_time_ms, file_name)

        # Case 3: Both Audio and Visual Present (Multimodal Video)
        if audio_telemetry is not None and visual_telemetry is not None:
            return self._fuse_both_modalities(audio_telemetry, visual_telemetry, processing_time_ms, file_name)

        # Fallback empty case
        return ConsolidatedForensicReport(
            media_type=media_type.upper(),
            verdict="REAL",
            overall_confidence=0.5,
            modality_breakdown={"audio": None, "visual": None, "classical_forensics": None},
            temporal_sync=[],
            top_anomalies=[],
            processing_time_ms=processing_time_ms,
            metadata={"file_name": file_name, "error": "No telemetry streams provided"},
        )

    # ── Fusion Subroutines ─────────────────────────────────────────────────────

    def _fuse_audio_only(
        self,
        audio_telemetry: Dict[str, Any],
        processing_time_ms: float,
        file_name: str,
    ) -> ConsolidatedForensicReport:
        verdict = str(audio_telemetry.get("verdict", "REAL"))
        confidence = float(audio_telemetry.get("confidence", 0.5))
        raw_scores = audio_telemetry.get("raw_scores", {})
        spoof_prob = float(raw_scores.get("spoof_prob", 0.5 if verdict == "FAKE" else 0.1))

        # Build temporal sync from audio chunks
        temporal_sync: List[Dict[str, Any]] = []
        audio_timeline = audio_telemetry.get("timeline", [])
        for chunk in audio_timeline:
            start = float(chunk.get("start_time_sec", 0.0))
            p_spoof = float(chunk.get("spoof_prob", spoof_prob))
            temporal_sync.append({
                "second": round(start, 2),
                "audio_spoof_prob": round(p_spoof, 4),
                "visual_spoof_prob": None,
                "fused_spoof_prob": round(p_spoof, 4),
                "is_anomaly": bool(p_spoof >= 0.55),
            })

        # Identify top anomalies
        top_anomalies = self._extract_audio_anomalies(audio_telemetry)

        modality_breakdown: Dict[str, Optional[Dict[str, Any]]] = {
            "audio": audio_telemetry,
            "visual": None,
            "classical_forensics": {
                "spectral_cues": audio_telemetry.get("spectral_cues", {}),
            },
        }

        return ConsolidatedForensicReport(
            media_type="AUDIO",
            verdict=verdict,
            overall_confidence=round(confidence, 4),
            modality_breakdown=modality_breakdown,
            temporal_sync=temporal_sync,
            top_anomalies=top_anomalies,
            processing_time_ms=processing_time_ms,
            metadata={
                "file_name": file_name,
                "raw_scores": raw_scores,
                "num_audio_chunks": len(audio_timeline),
            },
        )

    def _fuse_visual_only(
        self,
        visual_telemetry: Dict[str, Any],
        media_type: str,
        processing_time_ms: float,
        file_name: str,
    ) -> ConsolidatedForensicReport:
        verdict = str(visual_telemetry.get("verdict", "REAL"))
        confidence = float(visual_telemetry.get("confidence", 0.5))
        raw_scores = visual_telemetry.get("raw_scores", {})
        fake_prob = float(raw_scores.get("fake_prob", 0.5 if verdict == "FAKE" else 0.1))

        # Build temporal sync from visual timeline
        temporal_sync: List[Dict[str, Any]] = []
        visual_timeline = visual_telemetry.get("timeline", [])
        for frame_item in visual_timeline:
            t_sec = float(frame_item.get("timestamp_sec", 0.0))
            p_fake = float(frame_item.get("spoof_prob", fake_prob))
            temporal_sync.append({
                "second": round(t_sec, 3),
                "audio_spoof_prob": None,
                "visual_spoof_prob": round(p_fake, 4),
                "fused_spoof_prob": round(p_fake, 4),
                "is_anomaly": bool(p_fake >= 0.55),
            })

        top_anomalies = self._extract_visual_anomalies(visual_telemetry)

        modality_breakdown: Dict[str, Optional[Dict[str, Any]]] = {
            "audio": None,
            "visual": visual_telemetry,
            "classical_forensics": visual_telemetry.get("visual_cues", {}),
        }

        return ConsolidatedForensicReport(
            media_type=media_type,
            verdict=verdict,
            overall_confidence=round(confidence, 4),
            modality_breakdown=modality_breakdown,
            temporal_sync=temporal_sync,
            top_anomalies=top_anomalies,
            processing_time_ms=processing_time_ms,
            metadata={
                "file_name": file_name,
                "raw_scores": raw_scores,
                "num_frames": len(visual_timeline),
            },
        )

    def _fuse_both_modalities(
        self,
        audio_telemetry: Dict[str, Any],
        visual_telemetry: Dict[str, Any],
        processing_time_ms: float,
        file_name: str,
    ) -> ConsolidatedForensicReport:
        p_audio = float(audio_telemetry.get("raw_scores", {}).get("spoof_prob", 0.5))
        p_visual = float(visual_telemetry.get("raw_scores", {}).get("fake_prob", 0.5))

        # 1. Base weighted sum
        base_fused = (self.audio_weight * p_audio) + (self.video_weight * p_visual)

        # 2. Dynamic Gating & Strong Artifact Anomaly Boosting (>= 0.70)
        max_modality = max(p_audio, p_visual)
        if max_modality >= self.anomaly_threshold:
            # Boost overall spoof score if either modality detects strong manipulation artifacts
            fused_score = max(base_fused, 0.40 * (p_audio + p_visual) + 0.35 * max_modality)
        else:
            fused_score = base_fused

        fused_score = float(np.clip(fused_score, 0.01, 0.99))
        real_score = float(round(1.0 - fused_score, 4))

        if fused_score >= self.threshold:
            verdict = "FAKE"
            overall_confidence = fused_score
        else:
            verdict = "REAL"
            overall_confidence = real_score

        # 3. Cross-Modal Temporal Synchronization (Second-by-Second Alignment)
        temporal_sync = self._synchronize_timelines(
            audio_timeline=audio_telemetry.get("timeline", []),
            visual_timeline=visual_telemetry.get("timeline", []),
            fallback_audio=p_audio,
            fallback_visual=p_visual,
        )

        # 4. Top Anomalies Compilation
        top_anomalies = self._extract_multimodal_anomalies(
            audio_telemetry=audio_telemetry,
            visual_telemetry=visual_telemetry,
            temporal_sync=temporal_sync,
        )

        # Combined Classical Forensics
        classical = {
            **visual_telemetry.get("visual_cues", {}),
            "spectral_cues": audio_telemetry.get("spectral_cues", {}),
        }

        modality_breakdown: Dict[str, Optional[Dict[str, Any]]] = {
            "audio": audio_telemetry,
            "visual": visual_telemetry,
            "classical_forensics": classical,
        }

        return ConsolidatedForensicReport(
            media_type="MULTIMODAL",
            verdict=verdict,
            overall_confidence=round(overall_confidence, 4),
            modality_breakdown=modality_breakdown,
            temporal_sync=temporal_sync,
            top_anomalies=top_anomalies,
            processing_time_ms=processing_time_ms,
            metadata={
                "file_name": file_name,
                "fused_spoof_probability": round(fused_score, 4),
                "audio_spoof_prob": round(p_audio, 4),
                "visual_spoof_prob": round(p_visual, 4),
                "audio_weight": self.audio_weight,
                "video_weight": self.video_weight,
                "anomaly_boost_applied": bool(max_modality >= self.anomaly_threshold),
            },
        )

    # ── Temporal Alignment & Anomaly Ranking Helpers ───────────────────────────

    def _synchronize_timelines(
        self,
        audio_timeline: List[Dict[str, Any]],
        visual_timeline: List[Dict[str, Any]],
        fallback_audio: float,
        fallback_visual: float,
    ) -> List[Dict[str, Any]]:
        """Align second-by-second audio chunks and video frame probabilities."""
        # Find maximum duration
        max_a = max([float(c.get("end_time_sec", 0.0)) for c in audio_timeline], default=0.0)
        max_v = max([float(f.get("timestamp_sec", 0.0)) for f in visual_timeline], default=0.0)
        total_duration = max(max_a, max_v, 1.0)
        total_seconds = int(np.ceil(total_duration))

        synced: List[Dict[str, Any]] = []

        for sec in range(total_seconds):
            sec_start = float(sec)
            sec_end = float(sec + 1)

            # Find matching audio chunks
            a_probs = [
                float(c.get("spoof_prob", fallback_audio))
                for c in audio_timeline
                if float(c.get("start_time_sec", 0.0)) <= sec_end and float(c.get("end_time_sec", sec_start)) >= sec_start
            ]
            a_val = float(np.mean(a_probs)) if a_probs else fallback_audio

            # Find matching visual frames
            v_probs = [
                float(f.get("spoof_prob", fallback_visual))
                for f in visual_timeline
                if sec_start <= float(f.get("timestamp_sec", 0.0)) < sec_end
            ]
            v_val = float(np.mean(v_probs)) if v_probs else fallback_visual

            # Fused temporal score
            fused_t = (self.audio_weight * a_val) + (self.video_weight * v_val)
            if max(a_val, v_val) >= self.anomaly_threshold:
                fused_t = max(fused_t, 0.40 * (a_val + v_val) + 0.35 * max(a_val, v_val))
            fused_t = float(np.clip(fused_t, 0.01, 0.99))

            synced.append({
                "second": float(sec),
                "audio_spoof_prob": round(a_val, 4),
                "visual_spoof_prob": round(v_val, 4),
                "fused_spoof_prob": round(fused_t, 4),
                "is_anomaly": bool(fused_t >= 0.55),
            })

        return synced

    def _extract_audio_anomalies(self, audio_telemetry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract prioritized anomaly events from audio telemetry."""
        anomalies: List[Dict[str, Any]] = []
        cues = audio_telemetry.get("spectral_cues", {})

        for item in cues.get("peak_artifact_ranges", []):
            anomalies.append({
                "timestamp_sec": 0.0,
                "modality": "audio",
                "description": item.get("description", "Acoustic frequency anomaly detected"),
                "anomaly_score": 0.85 if item.get("severity") == "high" else 0.70,
            })

        for chunk in audio_telemetry.get("timeline", []):
            if float(chunk.get("spoof_prob", 0.0)) >= 0.60:
                anomalies.append({
                    "timestamp_sec": float(chunk.get("start_time_sec", 0.0)),
                    "modality": "audio",
                    "description": f"Audio chunk tampering detected (p={chunk.get('spoof_prob'):.2f})",
                    "anomaly_score": float(chunk.get("spoof_prob", 0.0)),
                })

        anomalies.sort(key=lambda x: x["anomaly_score"], reverse=True)
        return anomalies[:5]

    def _extract_visual_anomalies(self, visual_telemetry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract prioritized anomaly events from visual telemetry."""
        anomalies: List[Dict[str, Any]] = []
        artifacts = visual_telemetry.get("key_artifacts", [])

        for art in artifacts:
            p_spoof = float(art.get("spoof_prob", 0.0))
            if p_spoof >= 0.55:
                anomalies.append({
                    "timestamp_sec": float(art.get("timestamp_sec", 0.0)),
                    "modality": "visual",
                    "description": f"Facial artifact / boundary discontinuity detected at frame {art.get('frame_idx')}",
                    "anomaly_score": p_spoof,
                })

        cues = visual_telemetry.get("visual_cues", {})
        if float(cues.get("fft_spectral_anomaly", 0.0)) >= 0.65:
            anomalies.append({
                "timestamp_sec": 0.0,
                "modality": "visual",
                "description": "2D FFT periodic upsampling grid pattern detected in facial frequency spectrum",
                "anomaly_score": float(cues.get("fft_spectral_anomaly", 0.0)),
            })
        if float(cues.get("ela_discrepancy_score", 0.0)) >= 0.65:
            anomalies.append({
                "timestamp_sec": 0.0,
                "modality": "visual",
                "description": "Error Level Analysis: JPEG compression gradient discrepancy detected",
                "anomaly_score": float(cues.get("ela_discrepancy_score", 0.0)),
            })

        anomalies.sort(key=lambda x: x["anomaly_score"], reverse=True)
        return anomalies[:5]

    def _extract_multimodal_anomalies(
        self,
        audio_telemetry: Dict[str, Any],
        visual_telemetry: Dict[str, Any],
        temporal_sync: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract prioritized cross-modal and unimodal anomaly events."""
        anomalies = []
        anomalies.extend(self._extract_audio_anomalies(audio_telemetry))
        anomalies.extend(self._extract_visual_anomalies(visual_telemetry))

        # Check for cross-modal anomaly spikes in synchronized timeline
        for item in temporal_sync:
            if item.get("is_anomaly") and item.get("fused_spoof_prob", 0.0) >= 0.70:
                anomalies.append({
                    "timestamp_sec": float(item.get("second", 0.0)),
                    "modality": "cross_modal",
                    "description": f"Synchronized multimodal anomaly peak at second {int(item.get('second', 0))}",
                    "anomaly_score": float(item.get("fused_spoof_prob", 0.0)),
                })

        anomalies.sort(key=lambda x: x["anomaly_score"], reverse=True)
        return anomalies[:5]
