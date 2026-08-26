"""
Image deepfake analyzer — EfficientNet-B4 in single-frame mode.

Reuses the video model's spatial backbone to classify individual images.
Face detection (YuNet) is applied before classification.

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
from app.config.settings import settings
from app.utils.logger import get_logger
from app.video.models.efficientnet.model import EfficientNetB4Model
from app.video.preprocessing.face_cropper import FaceCropper
from app.video.preprocessing.face_detector import FaceDetector
from app.video.preprocessing.resolution_converter import ResolutionConverter
from app.video.preprocessing.video_normalizer import VideoNormalizer
from app.video.preprocessing.video_tensor_converter import VideoTensorConverter

logger = get_logger(__name__)

_THRESHOLD_FAKE = 0.70
_THRESHOLD_REAL = 0.30


class ImageAnalyzer:
    """Image deepfake analyzer using EfficientNet-B4 in single-frame mode.

    Uses the spatial backbone of the video deepfake detection model to
    analyze individual images. Face detection, cropping, and alignment
    are applied before classification.
    """

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
        """Load model and preprocessors on first call."""
        if self._model is not None:
            return

        logger.info("ImageAnalyzer: initialising model and preprocessors")

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
        """Analyze an image file for deepfake content.

        Parameters
        ----------
        file_path : str | Path
            Path to the image file.

        Returns
        -------
        AnalysisReport
        """
        start = time.perf_counter()
        self._ensure_model_loaded()
        file_path = Path(file_path)

        try:
            image = self._load_image(file_path)
            orig_h, orig_w = image.shape[:2]

            tensor, num_faces, bbox_info = self._preprocess_image(image)
            tensor = tensor.to(self.device)

            with torch.no_grad():
                logits = self._model(tensor)
                probs = F.softmax(logits, dim=-1)
                fake_prob = float(probs[0, 1].item())

            if fake_prob >= _THRESHOLD_FAKE:
                verdict = "FAKE"
            elif fake_prob <= _THRESHOLD_REAL:
                verdict = "REAL"
            else:
                verdict = "UNCERTAIN"

            elapsed = (time.perf_counter() - start) * 1000.0

            logger.info(
                "ImageAnalyzer: %s → %s (fake_prob=%.4f, faces=%d, %.1fms)",
                file_path.name, verdict, fake_prob, num_faces, elapsed,
            )

            metadata: Dict[str, Any] = {
                "file_name": file_path.name,
                "faces_detected": num_faces,
                "face_bbox": bbox_info,
                "original_dimensions": [orig_w, orig_h],
                "model": "EfficientNet-B4 (single-frame mode)",
                "input_resolution": "224x224",
            }

            return AnalysisReport(
                verdict=verdict,
                confidence=fake_prob,
                media_type="image",
                scores={"image": fake_prob},
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
                scores={"image": None},
                processing_time_ms=round(elapsed, 1),
                metadata={"error": str(exc), "file_name": file_path.name},
            )

    # ── Private helpers ───────────────────────

    def _load_image(self, file_path: Path) -> np.ndarray:
        """Load image via OpenCV and convert BGR → RGB."""
        image = cv2.imread(str(file_path))
        if image is None:
            raise ValueError(f"Failed to load image: {file_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, int, Optional[Dict[str, int]]]:
        """Detect face, crop, resize, convert to tensor, and normalise."""
        assert self._face_detector is not None

        face_box = self._face_detector.detect_largest(image)
        bbox_info: Optional[Dict[str, int]] = None

        if face_box is not None:
            bbox = (face_box.x, face_box.y, face_box.x + face_box.w, face_box.y + face_box.h)
            bbox_info = {"x": int(face_box.x), "y": int(face_box.y), "w": int(face_box.w), "h": int(face_box.h)}
            crop = self._face_cropper.crop(image, bbox=bbox)
            num_faces = 1
        else:
            crop = self._face_cropper.crop(image, bbox=None)
            num_faces = 0

        crop = self._resolution_converter.convert(crop)

        # VideoTensorConverter expects a list of frames
        tensor = self._tensor_converter.to_tensor([crop])        # [1, C, H, W]
        tensor = self._normalizer.normalize(tensor)

        # Model expects [B, T, C, H, W] — add batch dim if needed
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(0)  # [1, 1, C, H, W]

        return tensor, num_faces, bbox_info
