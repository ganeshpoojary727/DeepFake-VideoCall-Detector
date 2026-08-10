"""
Face detector for OpenCV 5+.

OpenCV 5 removed ``CascadeClassifier``.  This module uses two approaches:

1. **Primary**: ``cv2.FaceDetectorYN`` (YuNet DNN model) — downloads a small
   (~350KB) ONNX model from the OpenCV zoo on first use and caches it locally.
2. **Fallback**: Skin-colour segmentation for environments without internet
   access.  Lower accuracy but zero dependencies.

The cached model is stored in ``<project_root>/models_cache/yunet_face.onnx``.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)

# YuNet model URL (OpenCV GitHub release asset — ~340 KB)
_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
_CACHE_DIR = Path(__file__).resolve().parents[4] / "models_cache"
_YUNET_PATH = _CACHE_DIR / "face_detection_yunet_2023mar.onnx"


@dataclass
class FaceBox:
    """Bounding box for a detected face."""
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h


def _download_yunet() -> bool:
    """Download YuNet model if not cached. Returns True on success."""
    if _YUNET_PATH.exists():
        return True
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        logger.info("Downloading YuNet face detector (~340KB)...")
        urllib.request.urlretrieve(_YUNET_URL, _YUNET_PATH)
        logger.info("YuNet face detector downloaded to %s", _YUNET_PATH)
        return True
    except Exception as exc:
        logger.warning("Could not download YuNet model: %s", exc)
        return False


class _YuNetDetector:
    """Wrapper around cv2.FaceDetectorYN (OpenCV 5)."""

    def __init__(self, model_path: Path, conf_threshold: float = 0.6) -> None:
        if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        backend_id = getattr(cv2.dnn, "DNN_BACKEND_DEFAULT", 0)
        target_id = getattr(cv2.dnn, "DNN_TARGET_CPU", 0)
        self._det = cv2.FaceDetectorYN_create(
            str(model_path),
            "",
            (300, 300),
            score_threshold=conf_threshold,
            nms_threshold=0.3,
            top_k=5000,
            backend_id=backend_id,
            target_id=target_id,
        )

    def detect(self, image: np.ndarray, conf: float = 0.6) -> List[FaceBox]:
        h, w = image.shape[:2]
        self._det.setInputSize((w, h))
        ret, faces = self._det.detect(image)
        if ret is None or faces is None:
            return []
        boxes = []
        for face in faces:
            x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            # Clamp to image bounds
            x = max(0, x); y = max(0, y)
            fw = min(fw, w - x); fh = min(fh, h - y)
            if fw > 0 and fh > 0:
                boxes.append(FaceBox(x=x, y=y, w=fw, h=fh))
        return sorted(boxes, key=lambda b: b.area, reverse=True)


class _SkinDetector:
    """
    Fallback face detector using skin-colour segmentation.

    Detects skin-coloured blobs as face candidates.  Not suitable for
    production, but allows the pipeline to run without internet access.
    """

    def detect(self, image: np.ndarray) -> List[FaceBox]:
        # Convert to YCrCb for robust skin detection
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(
            ycrcb,
            np.array([0, 133, 77], dtype=np.uint8),
            np.array([255, 173, 127], dtype=np.uint8),
        )
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2000:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter non-face-like aspect ratios
            aspect = w / (h + 1e-6)
            if 0.4 < aspect < 2.0:
                boxes.append(FaceBox(x=x, y=y, w=w, h=h))
        return sorted(boxes, key=lambda b: b.area, reverse=True)


class FaceDetector:
    """
    Face detector compatible with OpenCV 5+.

    Tries YuNet (DNN-based, high accuracy) first; falls back to
    skin-colour segmentation if the model cannot be loaded.

    Parameters
    ----------
    conf_threshold : float
        YuNet confidence threshold (0.0–1.0).
    """

    def __init__(self, conf_threshold: float = 0.6) -> None:
        self._impl = None
        self._conf = conf_threshold

    def _init_detector(self) -> None:
        """Try YuNet, fall back to skin detector."""
        if self._impl is not None:
            return
        if _download_yunet():
            try:
                self._impl = _YuNetDetector(_YUNET_PATH, self._conf)
                logger.debug("FaceDetector: using YuNet (OpenCV 5 DNN)")
                return
            except Exception as exc:
                logger.warning("YuNet init failed: %s", exc)

        logger.info("FaceDetector: falling back to skin-colour segmentation")
        self._impl = _SkinDetector()

    def __getstate__(self) -> dict:
        """Exclude unpicklable C++ OpenCV FaceDetectorYN object during multiprocessing serialization."""
        state = self.__dict__.copy()
        state["_impl"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state for worker processes."""
        self.__dict__.update(state)
        self._impl = None

    # ── Public API ────────────────────────────

    def detect(self, image: np.ndarray) -> List[FaceBox]:
        """
        Detect all faces in a BGR image.

        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format (H, W, 3).

        Returns
        -------
        List[FaceBox]
            Detected face bounding boxes, sorted by area descending.
        """
        if self._impl is None:
            self._init_detector()

            return []
        try:
            return self._impl.detect(image)
        except Exception as exc:
            logger.warning("Face detection error: %s", exc)
            return []

    def detect_largest(self, image: np.ndarray) -> Optional[FaceBox]:
        """Return only the largest detected face, or ``None`` if none found."""
        boxes = self.detect(image)
        return boxes[0] if boxes else None

    def crop_face(
        self,
        image: np.ndarray,
        box: FaceBox,
        target_size: Tuple[int, int] = (224, 224),
        margin: float = 0.2,
    ) -> np.ndarray:
        """
        Crop and resize a face region from *image*.

        Parameters
        ----------
        image : np.ndarray
            Source BGR image.
        box : FaceBox
            Face bounding box.
        target_size : tuple[int, int]
            Output (width, height).
        margin : float
            Fractional margin added around the bounding box.

        Returns
        -------
        np.ndarray
            Cropped, resized face in BGR format.
        """
        h, w = image.shape[:2]
        margin_x = int(box.w * margin)
        margin_y = int(box.h * margin)

        x1 = max(0, box.x - margin_x)
        y1 = max(0, box.y - margin_y)
        x2 = min(w, box.x + box.w + margin_x)
        y2 = min(h, box.y + box.h + margin_y)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
