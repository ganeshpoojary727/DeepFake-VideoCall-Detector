"""
Image deepfake analyzer — Multi-Signal Ensemble (EfficientNet-B4 + Image Forensics Engine).

Combines:
1. Deep spatial convolutional neural network (EfficientNet-B4 + YuNet face detection)
2. 2D FFT Frequency analysis (detects periodic GAN upsampling artifacts)
3. Spatial Rich Model (SRM) sensor noise residuals (detects camera PRNU noise)
4. Error Level Analysis (ELA) (detects compression gradient discrepancies)
5. Chromatic dispersion consistency

Supports: .jpg, .jpeg, .png, .bmp, .webp, .tiff, .tif
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from app.analyzer.analysis_report import AnalysisReport
from app.analyzer.image_forensics import ImageForensics
from app.config.settings import settings
from app.utils.logger import get_logger
from app.video.models.efficientnet.model import EfficientNetB4Model
from app.video.preprocessing.face_cropper import FaceCropper
from app.video.preprocessing.face_detector import FaceDetector
from app.video.preprocessing.resolution_converter import ResolutionConverter
from app.video.preprocessing.video_normalizer import VideoNormalizer
from app.video.preprocessing.video_tensor_converter import VideoTensorConverter

logger = get_logger(__name__)

_DECISION_THRESHOLD = 0.65


class ImageAnalyzer:
    """Multi-Signal Image Deepfake Analyzer."""

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or settings.DEVICE
        self._model: Optional[EfficientNetB4Model] = None
        self._face_detector: Optional[FaceDetector] = None
        self._face_cropper: Optional[FaceCropper] = None
        self._resolution_converter: Optional[ResolutionConverter] = None
        self._tensor_converter: Optional[VideoTensorConverter] = None
        self._normalizer: Optional[VideoNormalizer] = None

    # ── Lazy initialisation ───────────────────

    def _ensure_model_loaded(self) -> None:
        """Load neural model and preprocessors on first call."""
        if self._model is not None:
            return

        logger.info("ImageAnalyzer: initialising neural model and preprocessors")
        self._model = EfficientNetB4Model()

        weights_path = settings.project_root / "trained_models" / "video" / "best_model.pt"
        if weights_path.exists():
            self._model.load_weights(str(weights_path), strict=False)
            logger.info("ImageAnalyzer: loaded weights from %s", weights_path)
        else:
            logger.warning("ImageAnalyzer: weights not found at %s", weights_path)

        self._model.set_mode("inference")
        self._model.to(self.device)
        self._model.eval()

        self._face_detector = FaceDetector(conf_threshold=0.6)
        self._face_cropper = FaceCropper(margin=0.2, target_size=(224, 224))
        self._resolution_converter = ResolutionConverter(target_resolution=(224, 224))
        self._tensor_converter = VideoTensorConverter(scale_to_unit=True)
        self._normalizer = VideoNormalizer()

    # ── Public API ────────────────────────────

    def analyze(self, file_path: str | Path) -> AnalysisReport:
        """Analyze an image file for deepfake content using multi-signal ensemble."""
        start = time.perf_counter()
        self._ensure_model_loaded()
        file_path = Path(file_path)

        try:
            image = self._load_image(file_path)
            orig_h, orig_w = image.shape[:2]

            # 1. Multi-Signal Physics & Frequency Forensics
            forensic_signals = ImageForensics.analyze_image_signals(image)
            forensic_fake_score = forensic_signals["combined_forensic_score"]

            # 2. Neural Feature Extraction
            tensor, num_faces, bbox_info = self._preprocess_image(image)
            tensor = tensor.to(self.device)

            with torch.no_grad():
                logits = self._model(tensor)
                probs = F.softmax(logits, dim=-1)
                neural_fake_score = float(probs[0, 1].item())

            # 3. Calibrated Multi-Signal Ensemble Fusion
            if num_faces > 0:
                # If face detected, balance spatial neural features with frequency forensics
                final_fake_prob = 0.65 * neural_fake_score + 0.35 * forensic_fake_score
            else:
                # If no face, rely more on full-image frequency/noise forensics
                final_fake_prob = 0.40 * neural_fake_score + 0.60 * forensic_fake_score

            final_fake_prob = float(np.clip(final_fake_prob, 0.01, 0.99))
            final_real_prob = float(round(1.0 - final_fake_prob, 4))

            # 4. Calibrated Verdict Selection
            if final_fake_prob >= _DECISION_THRESHOLD:
                verdict = "FAKE"
                verdict_confidence = final_fake_prob
            elif final_real_prob >= _DECISION_THRESHOLD:
                verdict = "REAL"
                verdict_confidence = final_real_prob
            else:
                verdict = "UNCERTAIN"
                verdict_confidence = max(final_real_prob, final_fake_prob)

            elapsed = (time.perf_counter() - start) * 1000.0

            logger.info(
                "ImageAnalyzer: %s → %s (Real=%.2f%%, Fake=%.2f%%, Neural=%.2f%%, Forensics=%.2f%%, %.1fms)",
                file_path.name, verdict, final_real_prob * 100, final_fake_prob * 100,
                neural_fake_score * 100, forensic_fake_score * 100, elapsed,
            )

            metadata: Dict[str, Any] = {
                "file_name": file_path.name,
                "faces_detected": num_faces,
                "face_bbox": bbox_info,
                "original_dimensions": [orig_w, orig_h],
                "neural_fake_probability": round(neural_fake_score, 4),
                "forensic_signals": forensic_signals,
                "model": "EfficientNet-B4 + Multi-Signal Forensic Engine",
                "input_resolution": "224x224",
            }

            return AnalysisReport(
                verdict=verdict,
                confidence=round(verdict_confidence, 4),
                media_type="image",
                real_confidence=round(final_real_prob, 4),
                fake_confidence=round(final_fake_prob, 4),
                scores={"image": round(final_fake_prob, 4)},
                processing_time_ms=round(elapsed, 1),
                metadata=metadata,
            )

        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000.0
            logger.exception("ImageAnalyzer: failed on %s: %s", file_path, exc)
            return AnalysisReport(
                verdict="UNCERTAIN",
                confidence=0.5,
                media_type="image",
                real_confidence=0.5,
                fake_confidence=0.5,
                scores={"image": None},
                processing_time_ms=round(elapsed, 1),
                metadata={"error": str(exc), "file_name": file_path.name},
            )

    # ── Private helpers ───────────────────────

    def _load_image(self, file_path: Path) -> np.ndarray:
        """Load image via OpenCV (BGR format)."""
        image = cv2.imread(str(file_path))
        if image is None:
            raise ValueError(f"Failed to load image: {file_path}")
        return image

    def _preprocess_image(self, image_bgr: np.ndarray) -> Tuple[torch.Tensor, int, Optional[Dict[str, int]]]:
        """Detect face, crop, resize, convert to RGB tensor, and normalise."""
        assert self._face_detector is not None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        face_box = self._face_detector.detect_largest(image_bgr)
        bbox_info: Optional[Dict[str, int]] = None

        if face_box is not None:
            bbox = (face_box.x, face_box.y, face_box.x + face_box.w, face_box.y + face_box.h)
            bbox_info = {"x": int(face_box.x), "y": int(face_box.y), "w": int(face_box.w), "h": int(face_box.h)}
            crop = self._face_cropper.crop(image_rgb, bbox=bbox)
            num_faces = 1
        else:
            crop = self._face_cropper.crop(image_rgb, bbox=None)
            num_faces = 0

        crop = self._resolution_converter.convert(crop)

        # VideoTensorConverter expects a list of frames
        tensor = self._tensor_converter.to_tensor([crop])        # [1, C, H, W]
        tensor = self._normalizer.normalize(tensor)

        # Model expects [B, T, C, H, W]
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(0)  # [1, 1, C, H, W]

        return tensor, num_faces, bbox_info
