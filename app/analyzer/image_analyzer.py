"""
Image Deepfake Analyzer — Multi-Signal Ensemble & Explainability.

Combines:
1. Deep spatial convolutional neural network (EfficientNet-B4 + Face Detection)
2. 2D FFT Frequency analysis (detects periodic GAN / Diffusion upsampling grid peaks)
3. Error Level Analysis (ELA) (detects JPEG compression discrepancy gradients)
4. Boundary Blending & Laplacian Inconsistency (detects face seam artifacts)
5. Grad-CAM visual explainability hotspots

Supports: .jpg, .jpeg, .png, .bmp, .webp, .tiff, .tif
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from app.analyzer.analysis_report import AnalysisReport
from app.analyzer.content_classifier import (
    ContentClassification,
    ContentClassifier,
    PHOTOGRAPHIC_HUMAN,
)
from app.analyzer.image_forensics import ImageForensics
from app.config.settings import settings
from app.utils.logger import get_logger
from app.video.models.efficientnet.model import EfficientNetB4Model
from app.video.preprocessing.face_cropper import FaceCropper
from app.video.preprocessing.face_detector import FaceDetector
from app.video.preprocessing.resolution_converter import ResolutionConverter
from app.video.preprocessing.video_normalizer import VideoNormalizer
from app.video.preprocessing.video_tensor_converter import VideoTensorConverter
from app.video.utils.visualization import GradCAM

logger = get_logger(__name__)

_THRESHOLD_FAKE = 0.55   # Confidence must be ≥55% to call FAKE
_THRESHOLD_REAL = 0.45   # Confidence must be ≤45% fake to call REAL
                         # 46–54% → UNCERTAIN (borderline coin-flip zone)

# Singleton Stage-0 content pre-classifier (lightweight, no GPU needed)
_content_classifier: Optional[ContentClassifier] = None


def _get_content_classifier() -> ContentClassifier:
    """Return the module-level ContentClassifier singleton (lazy-init)."""
    global _content_classifier
    if _content_classifier is None:
        _content_classifier = ContentClassifier(face_conf_threshold=0.50)
    return _content_classifier


class ImageAnalyzer:
    """Multi-Signal Image Deepfake Analyzer with Explainability Telemetry."""

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or settings.DEVICE
        self._model: Optional[EfficientNetB4Model] = None
        self._gradcam: Optional[GradCAM] = None
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

        logger.info("ImageAnalyzer: Initializing neural model and preprocessors")
        self._model = EfficientNetB4Model()

        video_dir = settings.project_root / "trained_models" / "video"
        weights_path = None
        for cand in ["best_accuracy.pt", "best_model.pt", "best_auc.pt"]:
            if (video_dir / cand).exists():
                weights_path = video_dir / cand
                break

        if weights_path is not None and weights_path.exists():
            self._model.load_weights(str(weights_path), strict=False)
            logger.info("ImageAnalyzer: Loaded weights from %s", weights_path)
        else:
            logger.warning("ImageAnalyzer: Weights not found in %s", video_dir)

        self._model.set_mode("inference")
        self._model.to(self.device)
        self._model.eval()

        self._gradcam = GradCAM(self._model)
        self._face_detector = FaceDetector(conf_threshold=0.6)
        self._face_cropper = FaceCropper(margin=0.2, target_size=(224, 224))
        self._resolution_converter = ResolutionConverter(target_resolution=(224, 224))
        self._tensor_converter = VideoTensorConverter(scale_to_unit=True)
        self._normalizer = VideoNormalizer()

    # ── Public APIs ───────────────────────────

    def analyze_structured(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Perform offline analysis on an uploaded image and return standardized telemetry schema.

        Returns
        -------
        Dict[str, Any]
            Standardized telemetry dictionary:
            - verdict: "REAL" | "FAKE" | "NOT_APPLICABLE"
            - confidence: float
            - raw_scores: {"real_prob": float, "fake_prob": float}
            - visual_cues: {"ela_discrepancy_score": float, "fft_spectral_anomaly": float, "boundary_inconsistency": float}
            - timeline: list with frame 0
            - key_artifacts: list of top anomalies with bbox and Grad-CAM saliency
            - content_classification: dict (Stage-0 pre-classifier result)
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        image_bgr = self._load_image(path)
        orig_h, orig_w = image_bgr.shape[:2]

        # ── Stage-0: Content Pre-Classification ──────────────────────────────
        clf = _get_content_classifier()
        content_result: ContentClassification = clf.classify(image_bgr)

        if not content_result.is_biometric_applicable:
            logger.info(
                "ImageAnalyzer: Stage-0 pre-classifier skipped deepfake analysis "
                "for %s — category=%s, confidence=%.3f",
                path.name,
                content_result.category,
                content_result.confidence,
            )
            return {
                "verdict": "NOT_APPLICABLE",
                "confidence": round(content_result.confidence, 4),
                "raw_scores": {
                    "real_prob": 0.0,
                    "fake_prob": 0.0,
                },
                "visual_cues": {
                    "ela_discrepancy_score": 0.0,
                    "fft_spectral_anomaly": 0.0,
                    "boundary_inconsistency": 0.0,
                },
                "timeline": [],
                "key_artifacts": [],
                "content_classification": content_result.to_dict(),
            }

        # Biometric content confirmed — load neural model and preprocessors
        self._ensure_model_loaded()

        face_box = self._face_detector.detect_largest(image_bgr)
        bbox_coords = None
        if face_box is not None:
            bbox_coords = (face_box.x, face_box.y, face_box.x + face_box.w, face_box.y + face_box.h)

        visual_cues = ImageForensics.extract_visual_cues(image_bgr, bbox=bbox_coords)
        forensic_fake_score = visual_cues["combined_score"]

        # 2. Neural Feature Extraction
        tensor, num_faces, bbox_info = self._preprocess_image(image_bgr, face_box)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = F.softmax(logits, dim=-1)
            neural_fake_score = float(probs[0, 1].item())

        # 3. Saliency Peak via Grad-CAM
        saliency_peak = [112, 112]
        if self._gradcam is not None:
            try:
                heatmap = self._gradcam.generate_heatmap(tensor, class_idx=1)
                px, py = GradCAM.extract_peak_saliency(heatmap)
                saliency_peak = [px, py]
            except Exception:
                saliency_peak = [112, 112]

        # 4. Multi-Signal Ensemble Fusion
        if num_faces > 0:
            final_fake_prob = 0.85 * neural_fake_score + 0.15 * forensic_fake_score
        else:
            final_fake_prob = 0.50 * neural_fake_score + 0.50 * forensic_fake_score

        final_fake_prob = float(np.clip(final_fake_prob, 0.01, 0.99))
        final_real_prob = float(round(1.0 - final_fake_prob, 4))

        # 5. Calibrated Verdict Determination
        # FAKE  → fake_prob ≥ 0.70  (definitive manipulation evidence)
        # REAL  → fake_prob ≤ 0.30  (definitive authenticity)
        # UNCERTAIN → 0.31–0.69   (coin-flip zone; never escalate)
        if final_fake_prob >= _THRESHOLD_FAKE:
            verdict = "FAKE"
            confidence = final_fake_prob
        elif final_fake_prob <= _THRESHOLD_REAL:
            verdict = "REAL"
            confidence = final_real_prob
        else:
            verdict = "UNCERTAIN"
            confidence = max(final_real_prob, final_fake_prob)

        # Format timeline & key artifacts
        timeline = [{
            "frame_idx": 0,
            "timestamp_sec": 0.0,
            "spoof_prob": round(final_fake_prob, 4),
            "is_anomaly": bool(final_fake_prob >= _THRESHOLD_FAKE),
        }]

        bbox_list = [bbox_info["x"], bbox_info["y"], bbox_info["x"] + bbox_info["w"], bbox_info["y"] + bbox_info["h"]] if bbox_info else [0, 0, orig_w, orig_h]
        key_artifacts = [{
            "frame_idx": 0,
            "timestamp_sec": 0.0,
            "bbox": bbox_list,
            "spoof_prob": round(final_fake_prob, 4),
            "saliency_peak": saliency_peak,
        }]

        # Include content_classification in the returned telemetry
        result = {
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "raw_scores": {
                "real_prob": round(final_real_prob, 4),
                "fake_prob": round(final_fake_prob, 4),
            },
            "visual_cues": {
                "ela_discrepancy_score": visual_cues["ela_discrepancy_score"],
                "fft_spectral_anomaly": visual_cues["fft_spectral_anomaly"],
                "boundary_inconsistency": visual_cues["boundary_inconsistency"],
            },
            "timeline": timeline,
            "key_artifacts": key_artifacts,
            "content_classification": content_result.to_dict(),
        }
        return result

    def analyze(self, file_path: Union[str, Path]) -> AnalysisReport:
        """Analyze an image file and return an enriched AnalysisReport."""
        start = time.perf_counter()
        path = Path(file_path)

        try:
            structured = self.analyze_structured(path)

            verdict = structured["verdict"]
            confidence = structured["confidence"]
            content_cls = structured.get("content_classification", {})

            # Stage-0 NOT_APPLICABLE short-circuit
            if verdict == "NOT_APPLICABLE":
                elapsed = (time.perf_counter() - start) * 1000.0
                reason = content_cls.get(
                    "reason",
                    "Non-biometric content; deepfake analysis not applicable.",
                )
                logger.info(
                    "ImageAnalyzer: %s → NOT_APPLICABLE — %s (%.1fms)",
                    path.name, reason, elapsed,
                )
                return AnalysisReport(
                    verdict="NOT_APPLICABLE",
                    confidence=round(confidence, 4),
                    media_type="image",
                    real_confidence=0.0,
                    fake_confidence=0.0,
                    scores={"image": None},
                    processing_time_ms=round(elapsed, 1),
                    content_category=content_cls.get("category"),
                    metadata={
                        "file_name": path.name,
                        "model": "Stage-0 ContentClassifier (no neural inference)",
                        "content_classification": content_cls,
                        "reason": reason,
                        "raw_scores": {"real_prob": 0.0, "fake_prob": 0.0},
                        "visual_cues": {
                            "ela_discrepancy_score": 0.0,
                            "fft_spectral_anomaly": 0.0,
                            "boundary_inconsistency": 0.0,
                        },
                        "timeline": [],
                        "key_artifacts": [],
                    },
                )

            real_prob = structured["raw_scores"]["real_prob"]
            fake_prob = structured["raw_scores"]["fake_prob"]
            visual_cues = structured["visual_cues"]
            key_artifacts = structured["key_artifacts"]

            elapsed = (time.perf_counter() - start) * 1000.0

            logger.info(
                "ImageAnalyzer: %s → %s (Real=%.2f%%, Fake=%.2f%%, %.1fms)",
                path.name, verdict, real_prob * 100, fake_prob * 100, elapsed,
            )

            metadata: Dict[str, Any] = {
                "file_name": path.name,
                "model": "EfficientNet-B4 + Multi-Signal Forensic Engine",
                "input_resolution": "224x224",
                "raw_scores": structured["raw_scores"],
                "visual_cues": visual_cues,
                "timeline": structured["timeline"],
                "key_artifacts": key_artifacts,
                "forensic_signals": ImageForensics.analyze_image_signals(self._load_image(path)),
            }

            return AnalysisReport(
                verdict=verdict,
                confidence=round(confidence, 4),
                media_type="image",
                real_confidence=round(real_prob, 4),
                fake_confidence=round(fake_prob, 4),
                scores={"image": round(fake_prob, 4)},
                processing_time_ms=round(elapsed, 1),
                metadata=metadata,
            )

        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000.0
            logger.exception("ImageAnalyzer: Failed on %s: %s", path, exc)
            return AnalysisReport(
                verdict="UNCERTAIN",
                confidence=0.5,
                media_type="image",
                real_confidence=0.5,
                fake_confidence=0.5,
                scores={"image": None},
                processing_time_ms=round(elapsed, 1),
                metadata={
                    "error": str(exc),
                    "file_name": path.name,
                    "raw_scores": {"real_prob": 0.5, "fake_prob": 0.5},
                    "visual_cues": {"ela_discrepancy_score": 0.5, "fft_spectral_anomaly": 0.5, "boundary_inconsistency": 0.5},
                    "timeline": [],
                    "key_artifacts": [],
                },
            )

    # ── Private helpers ───────────────────────

    def _load_image(self, file_path: Path) -> np.ndarray:
        """Load image via OpenCV (BGR format)."""
        image = cv2.imread(str(file_path))
        if image is None:
            raise ValueError(f"Failed to load image: {file_path}")
        return image

    def _preprocess_image(
        self,
        image_bgr: np.ndarray,
        face_box: Any = None,
    ) -> Tuple[torch.Tensor, int, Optional[Dict[str, int]]]:
        """Detect face, crop, resize, convert to RGB tensor, and normalise."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
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
        tensor = self._tensor_converter.to_tensor([crop])
        tensor = self._normalizer.normalize(tensor)

        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(0)  # [1, 1, C, H, W]

        return tensor, num_faces, bbox_info
