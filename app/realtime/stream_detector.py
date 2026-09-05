"""
Real-Time Video Call & Live Webcam Deepfake Detection Subsystem.

Provides low-latency temporal sliding-window inference, continuous face tracking,
Exponential Moving Average (EMA) confidence smoothing, and hysteresis thresholding
specifically engineered for live video calls (Zoom, Microsoft Teams, Google Meet)
and webcam feeds.
"""

from __future__ import annotations

import base64
import collections
import time
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from app.analyzer.content_classifier import ContentClassifier, PHOTOGRAPHIC_HUMAN
from app.analyzer.image_forensics import ImageForensics
from app.config.settings import settings
from app.utils.logger import get_logger
from app.video.face_detection.face_detector import FaceBox, FaceDetector
from app.video.inference.video_detector import VideoDeepfakeDetector
from app.video.inference.window_capture import WindowCapture
from app.video.preprocessing.face_cropper import FaceCropper

logger = get_logger(__name__)

# Hysteresis thresholds for live video calls
_THRESHOLD_FAKE = 0.70
_THRESHOLD_REAL = 0.30


class RealtimeLiveDetector:
    """High-performance real-time video stream analyzer for live webcam and video call streams."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        window_size: int = 16,
        ema_alpha: float = 0.82,
        history_maxlen: int = 60,
    ) -> None:
        self.device = device or settings.DEVICE
        self.window_size = window_size
        self.ema_alpha = ema_alpha

        # Preprocessing & primitives
        self.face_detector = FaceDetector(conf_threshold=0.55)
        self.face_cropper = FaceCropper(margin=0.2, target_size=(224, 224))
        self.window_buffer = WindowCapture(window_size=window_size, stride=2)
        self.content_classifier = ContentClassifier()
        self.forensics = ImageForensics()

        # Neural detector
        weights_path = Path(model_path or (settings.project_root / "trained_models" / "video" / "best_model.pt"))
        self.detector = VideoDeepfakeDetector(
            model_path=weights_path if weights_path.exists() else None,
            device=self.device,
            sequence_length=window_size,
        )

        # Temporal smoothing & hysteresis state
        self.smoothed_fake_prob = 0.50
        self.current_verdict = "UNCERTAIN"
        self.last_bbox: Optional[Tuple[int, int, int, int]] = None
        self.history_points: Deque[Dict[str, Any]] = collections.deque(maxlen=history_maxlen)

        # Performance counters
        self.frame_count = 0
        self.start_time = time.perf_counter()
        self.last_frame_time = time.perf_counter()
        self.fps_smoothed = 0.0

    def reset(self) -> None:
        """Reset internal frame buffers, temporal tracking, and history."""
        self.window_buffer.clear()
        self.smoothed_fake_prob = 0.50
        self.current_verdict = "UNCERTAIN"
        self.last_bbox = None
        self.history_points.clear()
        self.frame_count = 0
        self.start_time = time.perf_counter()
        self.last_frame_time = time.perf_counter()
        self.fps_smoothed = 0.0

    def process_base64_frame(
        self,
        b64_string: str,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Decode base64 image payload (e.g. from WebSocket or JSON POST) and process."""
        try:
            if "," in b64_string:
                b64_string = b64_string.split(",", 1)[1]
            raw_bytes = base64.b64decode(b64_string)
            nparr = np.frombuffer(raw_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Could not decode image from base64 data")
            return self.process_frame(frame, timestamp=timestamp)
        except Exception as exc:
            logger.error("Error decoding base64 live frame: %s", exc)
            return {
                "status": "error",
                "message": f"Frame decoding failed: {exc}",
                "verdict": "UNCERTAIN",
                "confidence": 0.5,
                "fake_confidence": 0.5,
                "real_confidence": 0.5,
                "face_detected": False,
                "fps": 0.0,
                "latency_ms": 0.0,
            }

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Process a single live video frame, update temporal state, and return telemetry.

        Parameters
        ----------
        frame : np.ndarray
            RGB or BGR image frame [H, W, 3]
        timestamp : Optional[float]
            Optional capture timestamp in seconds

        Returns
        -------
        Dict[str, Any]
            Real-time telemetry payload with verdict, confidence, bbox, FPS, and diagnostics.
        """
        t0 = time.perf_counter()
        now = timestamp or t0
        self.frame_count += 1

        # Calculate live FPS
        dt = t0 - self.last_frame_time
        self.last_frame_time = t0
        instant_fps = (1.0 / dt) if dt > 0.001 else 30.0
        self.fps_smoothed = (0.9 * self.fps_smoothed + 0.1 * instant_fps) if self.fps_smoothed > 0 else instant_fps

        h, w = frame.shape[:2]

        # 1. Stage-0 Content Check (avoid analyzing anime / illustrations)
        classification = self.content_classifier.classify(frame)
        if not classification.is_biometric_applicable:
            return {
                "status": "non_biometric",
                "verdict": "NOT_APPLICABLE",
                "confidence": round(classification.confidence, 4),
                "fake_confidence": 0.0,
                "real_confidence": 0.0,
                "face_detected": False,
                "bbox": None,
                "content_category": classification.category,
                "explanation": classification.reason or "Non-photographic or synthetic art detected; biometric deepfake forensics do not apply.",
                "fps": round(self.fps_smoothed, 1),
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                "history": list(self.history_points),
            }

        # 2. Face Detection
        faces = self.face_detector.detect(frame)
        face_detected = len(faces) > 0
        active_bbox: Optional[Dict[str, int]] = None
        face_crop: Optional[np.ndarray] = None

        if face_detected:
            # Pick primary (largest) face
            primary_face: FaceBox = faces[0]
            bx, by, bw, bh = primary_face.x, primary_face.y, primary_face.w, primary_face.h
            self.last_bbox = (bx, by, bw, bh)
            active_bbox = {"x": bx, "y": by, "w": bw, "h": bh}

            # Crop face for temporal buffer
            crop_coords = (bx, by, bx + bw, by + bh)
            face_crop = self.face_cropper.crop(frame, bbox=crop_coords)
        elif self.last_bbox is not None:
            # Fallback to last known face box with slight decay if tracking momentarily lost
            bx, by, bw, bh = self.last_bbox
            active_bbox = {"x": bx, "y": by, "w": bw, "h": bh}
            crop_coords = (bx, by, bx + bw, by + bh)
            face_crop = self.face_cropper.crop(frame, bbox=crop_coords)

        # 3. Add to temporal sliding window buffer
        if face_crop is not None:
            full_window = self.window_buffer.add_frame(face_crop)
        else:
            full_window = None

        # 4. Neural Model Inference on Temporal Window
        current_instant_fake = self.smoothed_fake_prob
        fast_cues = {"fft_spectral_anomaly": 0.1, "boundary_inconsistency": 0.1, "ela_discrepancy": 0.1}

        if full_window is not None and len(full_window) >= 8:
            try:
                # Use model to score the 16-frame window
                detailed = self.detector.predict_detailed(full_window, fps=self.fps_smoothed or 30.0)
                current_instant_fake = float(detailed["raw_scores"]["fake_prob"])
                fast_cues = detailed.get("visual_cues", fast_cues)
            except Exception as exc:
                logger.debug("Neural window inference fallback: %s", exc)
        elif face_crop is not None:
            # Fast single-frame fallback cue estimation when buffer is still building
            try:
                fft_val = float(self.forensics.compute_fft_azimuthal_discrepancy(face_crop))
                ela_val = float(self.forensics.compute_ela_discrepancy(face_crop))
                fast_cues["fft_spectral_anomaly"] = round(fft_val, 4)
                fast_cues["ela_discrepancy"] = round(ela_val, 4)
                instant_score = 0.5 * fft_val + 0.5 * ela_val
                # Gentle pull towards instant score while buffer fills
                current_instant_fake = 0.9 * self.smoothed_fake_prob + 0.1 * instant_score
            except Exception:
                pass

        # 5. Exponential Moving Average (EMA) Temporal Smoothing
        self.smoothed_fake_prob = float(
            self.ema_alpha * self.smoothed_fake_prob + (1.0 - self.ema_alpha) * current_instant_fake
        )
        self.smoothed_fake_prob = float(np.clip(self.smoothed_fake_prob, 0.01, 0.99))

        # 6. Hysteresis Decision Logic
        if self.smoothed_fake_prob >= _THRESHOLD_FAKE:
            self.current_verdict = "FAKE"
            confidence = self.smoothed_fake_prob
        elif self.smoothed_fake_prob <= _THRESHOLD_REAL:
            self.current_verdict = "REAL"
            confidence = 1.0 - self.smoothed_fake_prob
        else:
            self.current_verdict = "UNCERTAIN"
            confidence = 1.0 - 2.0 * abs(self.smoothed_fake_prob - 0.5)

        # 7. Threat Level Assessment
        if self.current_verdict == "FAKE":
            threat_level = "CRITICAL"
        elif self.smoothed_fake_prob >= 0.50:
            threat_level = "ELEVATED"
        else:
            threat_level = "NOMINAL"

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # 8. Record telemetry history point
        point = {
            "t": round(now - self.start_time, 2),
            "score": round(self.smoothed_fake_prob, 3),
            "verdict": self.current_verdict,
            "fps": round(self.fps_smoothed, 1),
        }
        self.history_points.append(point)

        return {
            "status": "active",
            "verdict": self.current_verdict,
            "confidence": round(confidence, 4),
            "fake_confidence": round(self.smoothed_fake_prob, 4),
            "real_confidence": round(1.0 - self.smoothed_fake_prob, 4),
            "face_detected": face_detected,
            "bbox": active_bbox,
            "threat_level": threat_level,
            "buffer_depth": len(self.window_buffer._buffer),
            "buffer_capacity": self.window_size,
            "fps": round(self.fps_smoothed, 1),
            "latency_ms": round(latency_ms, 1),
            "visual_cues": fast_cues,
            "history": list(self.history_points),
        }
