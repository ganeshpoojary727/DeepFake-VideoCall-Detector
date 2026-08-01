"""
Video deepfake detector — face extraction + model inference + heuristic fallback.

Pipeline per detection call
---------------------------
1. Receive BGR frame(s) from ScreenCapture or a video file
2. Detect + crop face ROI via OpenCV
3. Analyze spatial Laplacian variance, high-frequency noise (DCT), face motion
4. Return a probability score [0.0 to 1.0]
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Deque, Iterator, List, Optional

import numpy as np
import torch

from app.config.settings import settings
from app.core.interfaces import BaseDetector, DetectionLabel, Modality, PredictionResult
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# VideoDeepfakeDetector — heuristic-based frame analysis
# ──────────────────────────────────────────────


class VideoDeepfakeDetector:
    """
    Video deepfake detector using signal-processing heuristics.

    Does NOT require a trained model — analyzes spatial texture,
    frequency artifacts, and temporal face motion consistency to
    estimate a fake probability.

    Analysis Pipeline
    -----------------
    1. Detect face ROI in each frame
    2. Compute per-frame Laplacian variance (blur detection)
    3. Compute per-frame high-frequency noise via DCT
    4. Assess temporal face position consistency
    5. Aggregate signals into a fake probability [0, 1]
    """

    def __init__(self) -> None:
        import cv2

        self._face_cascade = None
        self._face_detector = None
        self._detection_method = "fullframe"

        # Try CascadeClassifier
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if Path(cascade_path).exists():
                cascade = cv2.CascadeClassifier(cascade_path)
                if hasattr(cascade, "empty") and not cascade.empty():
                    self._face_cascade = cascade
                    self._detection_method = "haar"
                    logger.info("VideoDeepfakeDetector: initialized with Haar cascade")
        except Exception as exc:
            logger.debug("Haar cascade init skipped: %s", exc)

        if self._detection_method == "fullframe":
            logger.info("VideoDeepfakeDetector: using center-crop spatial analysis")

    def predict_from_frames(self, frames_list: List[np.ndarray]) -> float:
        """
        Predict deepfake probability from a list of BGR frames.

        Parameters
        ----------
        frames_list : list[np.ndarray]
            List of BGR frames (H, W, 3) from the video buffer.

        Returns
        -------
        float
            Fake probability from 0.0 (definitely real) to 1.0 (definitely fake).
        """
        import cv2

        if not frames_list or len(frames_list) < 3:
            logger.debug("Too few frames for video analysis (%d)", len(frames_list))
            return 0.5

        start = time.perf_counter()

        # Sub-sample frames for efficiency (analyze up to 25 frames)
        step = max(1, len(frames_list) // 25)
        sampled = frames_list[::step][:25]

        laplacian_vars = []
        hf_noise_scores = []
        face_centers = []
        face_found_count = 0

        for frame in sampled:
            if frame is None or frame.size == 0:
                continue

            # Convert to grayscale
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            h, w = gray.shape[:2]
            face_roi = None
            cx, cy = w // 2, h // 2

            # Strategy 1: Haar Cascade if available
            if self._detection_method == "haar" and self._face_cascade is not None:
                try:
                    faces = self._face_cascade.detectMultiScale(
                        gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30)
                    )
                    if len(faces) > 0:
                        largest = max(faces, key=lambda f: f[2] * f[3])
                        fx, fy, fw, fh = largest
                        face_roi = gray[fy:fy + fh, fx:fx + fw]
                        cx, cy = fx + fw // 2, fy + fh // 2
                        face_found_count += 1
                except Exception:
                    pass

            # Strategy 2: Center crop fallback
            if face_roi is None or face_roi.size == 0:
                crop_size = min(h, w) // 2
                if crop_size > 20:
                    y1 = max(0, cy - crop_size // 2)
                    y2 = min(h, cy + crop_size // 2)
                    x1 = max(0, cx - crop_size // 2)
                    x2 = min(w, cx + crop_size // 2)
                    face_roi = gray[y1:y2, x1:x2]
                    face_found_count += 1

            if face_roi is not None and face_roi.size > 0:
                face_centers.append((cx, cy))

                # --- Laplacian Variance (blur / sharpness) ---
                lap = cv2.Laplacian(face_roi, cv2.CV_64F)
                laplacian_vars.append(float(lap.var()))

                # --- High-Frequency Noise (DCT analysis) ---
                if face_roi.shape[0] >= 8 and face_roi.shape[1] >= 8:
                    face_resized = cv2.resize(face_roi, (64, 64)).astype(np.float32)
                    dct_coeff = cv2.dct(face_resized)
                    # High-frequency energy (bottom-right quadrant of DCT)
                    hf_block = dct_coeff[32:, 32:]
                    hf_energy = float(np.mean(np.abs(hf_block)))
                    total_energy = float(np.mean(np.abs(dct_coeff))) + 1e-10
                    hf_noise_scores.append(hf_energy / total_energy)

        elapsed = (time.perf_counter() - start) * 1000

        # --- Aggregate Signals ---
        fake_score = 0.0

        # Signal 1: Laplacian variance (unusually uniform blur = potential deepfake)
        if laplacian_vars and len(laplacian_vars) >= 3:
            mean_lap = np.mean(laplacian_vars)
            std_lap = np.std(laplacian_vars)
            cv_lap = std_lap / (mean_lap + 1e-10)  # Coefficient of variation

            if mean_lap < 40:
                fake_score += 0.2
            if cv_lap < 0.1:  # Unusually consistent blur
                fake_score += 0.15

        # Signal 2: High-frequency noise consistency
        if hf_noise_scores and len(hf_noise_scores) >= 3:
            mean_hf = np.mean(hf_noise_scores)
            std_hf = np.std(hf_noise_scores)

            if std_hf < 0.01:
                fake_score += 0.15
            if mean_hf > 0.3:
                fake_score += 0.1

        # Signal 3: Temporal motion consistency
        if face_centers and len(face_centers) >= 5:
            centers = np.array(face_centers, dtype=np.float32)
            diffs = np.diff(centers, axis=0)
            motion_magnitudes = np.sqrt(np.sum(diffs ** 2, axis=1))
            motion_std = np.std(motion_magnitudes)

            if motion_std < 0.5 and np.mean(motion_magnitudes) < 1.5:
                fake_score += 0.1

        # Normalize score into [0.1, 0.9] range
        fake_score = float(np.clip(fake_score + 0.1, 0.0, 1.0))

        logger.debug(
            "VideoDeepfakeDetector: score=%.4f, frames=%d/%d, latency=%.1fms",
            fake_score, face_found_count, len(sampled), elapsed,
        )
        return fake_score

    @property
    def is_ready(self) -> bool:
        """Always ready — uses heuristic analysis."""
        return True


# Backward compatibility alias
VideoDetector = VideoDeepfakeDetector