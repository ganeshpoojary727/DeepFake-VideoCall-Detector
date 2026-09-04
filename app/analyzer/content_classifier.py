"""
Stage-0 Content Pre-Classifier for DeepFake-VideoCall-Detector.

Rapidly classifies incoming images into one of three categories BEFORE any
biometric deepfake analysis is attempted.  The classifier is intentionally
lightweight (sub-10 ms on CPU) and uses only OpenCV, NumPy, and PIL.

Categories
----------
PHOTOGRAPHIC_HUMAN  : Photorealistic image with a human subject.
DIGITAL_ART_ANIME   : Cartoon, anime, illustration, vector art, painting.
SCENERY_OBJECT      : Landscape, architecture, animal, vehicle, or abstract
                      wallpaper without a human face.

Detection cues (multi-signal fusion)
-------------------------------------
1. Edge-to-Texture & Line-Art Ratio
   Cartoons and anime exhibit high-density hard Canny edges relative to the
   overall texture gradient variance.  Real photos have continuous, smooth
   Poisson-Gaussian texture gradients with fewer isolated hard edges.

2. Color Palette Entropy & Flatness
   Digital art uses quantized, flat-shaded color clusters (low intra-cluster
   variance).  Photographs show rich continuous color gradients with high
   palette entropy.

3. Face Anthropometry (landmark-free, shape-based)
   When a candidate face region is detected, the aspect ratio and relative
   face area are evaluated against natural human anthropometric priors.
   Stylised anime faces often violate these geometric constraints.

4. Sensor Noise (PRNU) Absence Test
   Physical camera images retain a measurable high-frequency Photo-Response
   Non-Uniformity (PRNU) noise floor.  Digitally generated artwork lacks
   this characteristic stochastic noise pattern entirely.

Usage
-----
    from app.analyzer.content_classifier import ContentClassifier

    clf = ContentClassifier()
    result = clf.classify(image_bgr)   # np.ndarray (BGR)
    # or
    result = clf.classify_file("path/to/image.jpg")

    if not result.is_biometric_applicable:
        # Skip deepfake analysis -- return NOT_APPLICABLE verdict
        ...
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Category constants
# ---------------------------------------------------------------------------

PHOTOGRAPHIC_HUMAN = "PHOTOGRAPHIC_HUMAN"
DIGITAL_ART_ANIME = "DIGITAL_ART_ANIME"
SCENERY_OBJECT = "SCENERY_OBJECT"

# Working resolution for all feature extraction (speed vs. accuracy tradeoff)
_ANALYSIS_SIZE = (256, 256)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ContentClassification:
    """Result returned by :class:`ContentClassifier`.

    Attributes
    ----------
    category : str
        One of ``"PHOTOGRAPHIC_HUMAN"``, ``"DIGITAL_ART_ANIME"``, or
        ``"SCENERY_OBJECT"``.
    is_biometric_applicable : bool
        ``True`` only for ``PHOTOGRAPHIC_HUMAN``.  When ``False``, deepfake
        analysis must be skipped and a ``NOT_APPLICABLE`` verdict returned.
    confidence : float
        Classifier confidence in [0.0, 1.0].
    reason : str
        Human-readable explanation of the classification decision.
    details : dict
        Per-cue signal values for diagnostics and logging.
    elapsed_ms : float
        Wall-clock time taken by the classifier (milliseconds).
    """

    category: str
    is_biometric_applicable: bool
    confidence: float
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain JSON-compatible dictionary."""
        return {
            "category": self.category,
            "is_biometric_applicable": self.is_biometric_applicable,
            "confidence": round(float(self.confidence), 4),
            "reason": self.reason,
            "details": self.details,
            "elapsed_ms": round(float(self.elapsed_ms), 2),
        }

    @property
    def verdict_label(self) -> str:
        """Return the NOT_APPLICABLE / APPLICABLE verdict string."""
        return "NOT_APPLICABLE" if not self.is_biometric_applicable else "APPLICABLE"


# ---------------------------------------------------------------------------
# Main Classifier
# ---------------------------------------------------------------------------

class ContentClassifier:
    """Lightweight Stage-0 content pre-classifier.

    All public methods accept either a file path or a pre-loaded BGR NumPy
    array so the classifier can be slotted into any point in the pipeline.

    Parameters
    ----------
    face_conf_threshold : float
        YuNet face-detection confidence threshold used during the
        anthropometry cue.  A lower value catches more faces (including
        potentially stylised ones).
    """

    # ------------------------------------------------------------------
    # Thresholds (tuned empirically across photographic vs. art datasets)
    # ------------------------------------------------------------------

    # If the aggregate digital-art score exceeds this, classify as DIGITAL_ART_ANIME
    _ART_SCORE_THRESHOLD: float = 0.52

    # Minimum art-vs-photo score margin required to confirm art classification
    _ART_MARGIN: float = 0.06

    # ------------------------------------------------------------------
    # Cue weights for the ensemble fusion
    # ------------------------------------------------------------------

    _W_EDGE_LINE_ART: float = 0.30     # Cue 1: edge-to-texture / line-art ratio
    _W_PALETTE_ENTROPY: float = 0.28   # Cue 2: colour palette entropy & flatness
    _W_ANTHROPOMETRY: float = 0.22     # Cue 3: face shape anthropometry
    _W_PRNU_NOISE: float = 0.20        # Cue 4: PRNU sensor-noise absence

    def __init__(self, face_conf_threshold: float = 0.50) -> None:
        self._face_conf = face_conf_threshold
        # Lazy-loaded to avoid YuNet download at import time
        self._face_detector: Optional[Any] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, image_bgr: np.ndarray) -> ContentClassification:
        """Classify a BGR NumPy image array.

        Parameters
        ----------
        image_bgr : np.ndarray
            OpenCV BGR image (H, W, 3) or (H, W) grayscale.

        Returns
        -------
        ContentClassification
            Structured classification result.
        """
        t0 = time.perf_counter()
        result = self._run_classification(image_bgr)
        result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.debug(
            "ContentClassifier: %s (confidence=%.3f, %.1fms) — %s",
            result.category,
            result.confidence,
            result.elapsed_ms,
            result.reason,
        )
        return result

    def classify_file(self, file_path: Union[str, Path]) -> ContentClassification:
        """Load an image from *file_path* and classify it.

        Parameters
        ----------
        file_path : str | Path
            Absolute or relative path to the image file.

        Returns
        -------
        ContentClassification

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file cannot be decoded as an image.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(
                f"ContentClassifier: image not found: {path}"
            )
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            raise ValueError(
                f"ContentClassifier: failed to decode image: {path}"
            )
        return self.classify(image_bgr)

    # ------------------------------------------------------------------
    # Core classification logic
    # ------------------------------------------------------------------

    def _run_classification(
        self, image_bgr: np.ndarray
    ) -> ContentClassification:
        """Execute multi-cue feature extraction and fuse scores."""
        if image_bgr is None or image_bgr.size == 0:
            return ContentClassification(
                category=SCENERY_OBJECT,
                is_biometric_applicable=False,
                confidence=0.90,
                reason="Empty or null image array received.",
                details={},
            )

        # Ensure 3-channel BGR
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        elif image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

        # Downsample to working resolution for speed
        small = cv2.resize(image_bgr, _ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)

        # ── Cue 1: Edge-to-Texture & Line-Art Ratio ───────────────────────────
        c1_art_score, c1_details = self._cue_edge_line_art(small)

        # ── Cue 2: Color Palette Entropy & Flatness ───────────────────────────
        c2_art_score, c2_details = self._cue_palette_entropy(small)

        # ── Cue 3: Face Anthropometry ─────────────────────────────────────────
        face_boxes = self._detect_faces(image_bgr)
        c3_art_score, c3_details = self._cue_face_anthropometry(
            image_bgr, face_boxes
        )

        # ── Cue 4: PRNU Sensor-Noise Absence ─────────────────────────────────
        c4_art_score, c4_details = self._cue_prnu_noise(small)

        # ── Weighted ensemble ─────────────────────────────────────────────────
        art_score = (
            self._W_EDGE_LINE_ART * c1_art_score
            + self._W_PALETTE_ENTROPY * c2_art_score
            + self._W_ANTHROPOMETRY * c3_art_score
            + self._W_PRNU_NOISE * c4_art_score
        )
        art_score = float(np.clip(art_score, 0.0, 1.0))
        photo_score = 1.0 - art_score

        details: Dict[str, Any] = {
            "art_score": round(art_score, 4),
            "photo_score": round(photo_score, 4),
            "face_count": len(face_boxes),
            "cue_1_edge_line_art": c1_details,
            "cue_2_palette_entropy": c2_details,
            "cue_3_face_anthropometry": c3_details,
            "cue_4_prnu_noise": c4_details,
        }

        has_face = len(face_boxes) > 0

        # ── Decision tree ─────────────────────────────────────────────────────
        if art_score >= self._ART_SCORE_THRESHOLD and (
            art_score - photo_score >= self._ART_MARGIN
        ):
            # Clear digital-art signal regardless of face presence
            return ContentClassification(
                category=DIGITAL_ART_ANIME,
                is_biometric_applicable=False,
                confidence=round(float(np.clip(art_score, 0.50, 0.99)), 4),
                reason=self._build_art_reason(
                    c1_details, c2_details, c3_details, c4_details
                ),
                details=details,
            )

        if not has_face:
            # No human face detected
            if art_score >= 0.40:
                return ContentClassification(
                    category=DIGITAL_ART_ANIME,
                    is_biometric_applicable=False,
                    confidence=round(float(np.clip(art_score, 0.50, 0.99)), 4),
                    reason=(
                        "No human face detected and the digital-art signal is elevated. "
                        "Image appears to be non-photographic artwork or a wallpaper."
                    ),
                    details=details,
                )
            return ContentClassification(
                category=SCENERY_OBJECT,
                is_biometric_applicable=False,
                confidence=round(float(np.clip(photo_score, 0.50, 0.99)), 4),
                reason=(
                    "No human face detected in the image. "
                    "Content appears to be scenery, an object, or a non-human subject. "
                    "Deepfake biometric analysis is not applicable to this content type."
                ),
                details=details,
            )

        # Face detected + image looks photographic
        return ContentClassification(
            category=PHOTOGRAPHIC_HUMAN,
            is_biometric_applicable=True,
            confidence=round(float(np.clip(photo_score, 0.50, 0.99)), 4),
            reason=(
                f"Photorealistic human face detected (face_count={len(face_boxes)}). "
                "Image passes the biometric applicability check; proceeding with "
                "deepfake analysis."
            ),
            details=details,
        )

    # ------------------------------------------------------------------
    # Cue 1: Edge-to-Texture & Line-Art Ratio
    # ------------------------------------------------------------------

    @staticmethod
    def _cue_edge_line_art(
        small: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute the line-art / edge density ratio relative to texture richness.

        Cartoons and anime: many isolated thin-stroke Canny edges paired with
        large flat colour-fill regions → high line-art index.
        Real photographs: continuous Poisson-Gaussian texture gradients,
        moderate Canny edge density.

        Returns
        -------
        art_score : float  [0 = photographic, 1 = digital art]
        details : dict
        """
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Canny edge density
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)

        # Sobel gradient magnitude variance (texture richness)
        gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.hypot(gx, gy)
        texture_variance = float(np.var(grad_mag))

        # Flat-region ratio: fraction of pixels with gradient < 5 % of maximum
        grad_max = float(np.max(grad_mag)) if np.max(grad_mag) > 0 else 1.0
        flat_ratio = float(np.mean(grad_mag < 0.05 * grad_max))

        # Line-art index: high edge density + large flat areas + low texture var
        normalized_tv = float(np.clip(texture_variance / 5000.0, 0.0, 1.0))
        line_art_index = (
            0.40 * edge_density
            + 0.35 * flat_ratio
            + 0.25 * (1.0 - normalized_tv)
        )

        # Sigmoid with inflection at line_art_index ~ 0.38
        art_score = float(
            1.0 / (1.0 + np.exp(-10.0 * (line_art_index - 0.38)))
        )
        art_score = float(np.clip(art_score, 0.02, 0.98))

        return art_score, {
            "edge_density": round(edge_density, 4),
            "texture_variance": round(texture_variance, 2),
            "flat_ratio": round(flat_ratio, 4),
            "line_art_index": round(line_art_index, 4),
            "art_score": round(art_score, 4),
        }

    # ------------------------------------------------------------------
    # Cue 2: Color Palette Entropy & Flatness
    # ------------------------------------------------------------------

    @staticmethod
    def _cue_palette_entropy(
        small: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        """Measure how quantised and flat-shaded the image's colour palette is.

        Digital art: small number of distinct, pure hue clusters → low palette
        entropy, low intra-cluster variance.
        Photographs: continuous gamut, high entropy.

        Approach: quantise to k=16 colours via k-means in Lab space; measure
        intra-cluster compactness and histogram entropy.

        Returns
        -------
        art_score : float
        details : dict
        """
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2Lab).astype(np.float32)
        pixels = lab.reshape(-1, 3)

        k = 16
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            20,
            0.5,
        )
        try:
            _, labels, _ = cv2.kmeans(
                pixels,
                k,
                None,
                criteria,
                attempts=3,
                flags=cv2.KMEANS_PP_CENTERS,
            )
        except Exception:
            return 0.5, {"error": "kmeans_failed", "art_score": 0.5}

        labels = labels.flatten()

        # Intra-cluster variance (flatness indicator)
        intra_vars: List[float] = []
        for cluster_id in range(k):
            mask = labels == cluster_id
            if int(np.sum(mask)) < 5:
                continue
            cluster_pts = pixels[mask]
            intra_vars.append(float(np.mean(np.var(cluster_pts, axis=0))))

        mean_intra_var = float(np.mean(intra_vars)) if intra_vars else 100.0

        # Cluster population entropy
        counts = np.bincount(labels, minlength=k).astype(np.float64)
        probs = counts / (counts.sum() + 1e-9)
        entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
        max_entropy = float(np.log(k))
        normalized_entropy = float(np.clip(entropy / max_entropy, 0.0, 1.0))

        # Flatness score: 0 intra-var = perfectly flat; >= 80 = noisy photo
        flatness_score = float(np.clip(1.0 - mean_intra_var / 80.0, 0.0, 1.0))

        # HSV saturation variance (vivid anime vs. washed watercolour)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat = hsv[:, :, 1].flatten() / 255.0
        sat_variance = float(np.var(sat))
        sat_art_signal = float(np.clip(sat_variance * 3.0, 0.0, 1.0))

        # Low entropy + high flatness -> digital art
        low_entropy_signal = 1.0 - normalized_entropy
        art_score = (
            0.45 * flatness_score
            + 0.35 * low_entropy_signal
            + 0.20 * sat_art_signal
        )
        art_score = float(np.clip(art_score, 0.02, 0.98))

        return art_score, {
            "mean_intra_cluster_variance": round(mean_intra_var, 2),
            "palette_entropy": round(entropy, 4),
            "normalized_entropy": round(normalized_entropy, 4),
            "flatness_score": round(flatness_score, 4),
            "saturation_variance": round(sat_variance, 4),
            "art_score": round(art_score, 4),
        }

    # ------------------------------------------------------------------
    # Cue 3: Face Anthropometry
    # ------------------------------------------------------------------

    @staticmethod
    def _cue_face_anthropometry(
        image_bgr: np.ndarray,
        face_boxes: list,
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate whether detected face regions exhibit natural human proportions.

        Human faces: width/height in [0.60, 0.95]; face occupies 3–40 % of image.
        Anime/cartoon faces: often violate these aspect-ratio and area-ratio priors.

        Returns
        -------
        art_score : float  [0 = photographic, 1 = stylised]
        details : dict
        """
        if not face_boxes:
            # No face detected — neutral cue (does not push toward art)
            return 0.35, {"reason": "no_face_detected"}

        h_img, w_img = image_bgr.shape[:2]
        image_area = float(h_img * w_img)

        art_scores: List[float] = []
        face_details: List[Dict[str, Any]] = []

        for fb in face_boxes[:3]:   # assess at most 3 largest faces
            fw, fh = float(fb.w), float(fb.h)
            if fw < 5 or fh < 5:
                continue

            aspect_ratio = fw / (fh + 1e-6)
            face_area_ratio = (fw * fh) / (image_area + 1e-6)

            # Aspect ratio deviation from natural range [0.60, 0.95]
            if 0.60 <= aspect_ratio <= 0.95:
                aspect_art = 0.0
            else:
                deviation = min(
                    abs(aspect_ratio - 0.60),
                    abs(aspect_ratio - 0.95),
                )
                aspect_art = float(np.clip(deviation / 0.40, 0.0, 1.0))

            # Face-area ratio deviation from natural range [0.03, 0.40]
            if 0.03 <= face_area_ratio <= 0.40:
                area_art = 0.0
            else:
                area_dev = min(
                    abs(face_area_ratio - 0.03),
                    abs(face_area_ratio - 0.40),
                )
                area_art = float(np.clip(area_dev / 0.25, 0.0, 1.0))

            face_art = 0.60 * aspect_art + 0.40 * area_art
            art_scores.append(face_art)
            face_details.append({
                "aspect_ratio": round(aspect_ratio, 3),
                "face_area_ratio": round(face_area_ratio, 4),
                "aspect_art_signal": round(aspect_art, 4),
                "area_art_signal": round(area_art, 4),
                "face_art_score": round(face_art, 4),
            })

        if not art_scores:
            return 0.35, {"reason": "face_too_small"}

        mean_art = float(np.mean(art_scores))
        return float(np.clip(mean_art, 0.02, 0.98)), {
            "faces_evaluated": len(art_scores),
            "per_face": face_details,
            "art_score": round(mean_art, 4),
        }

    # ------------------------------------------------------------------
    # Cue 4: PRNU Sensor-Noise Absence
    # ------------------------------------------------------------------

    @staticmethod
    def _cue_prnu_noise(
        small: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        """Measure the high-frequency PRNU-like noise floor.

        Real photographs captured by a physical sensor contain a measurable
        stochastic noise residual in the high-frequency band.  Digitally
        synthesised artwork lacks this pattern, yielding near-zero high-pass
        residual variance.

        Method: apply a Laplacian high-pass filter and measure per-patch
        variance across a 16 x 16 grid.  Low mean patch variance -> digital art.

        Calibration
        -----------
        mean_var < 1.5   -> almost no noise     -> very likely digital art
        mean_var 1.5-6   -> low noise            -> possibly rendered / CG
        mean_var 6-20    -> moderate noise       -> leaning photographic
        mean_var 20-70   -> healthy photo range  -> photographic
        mean_var > 70    -> heavy noise          -> extreme ISO / JPEG artefact

        Returns
        -------
        art_score : float
        details : dict
        """
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

        kernel = np.array(
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32
        )
        residual = cv2.filter2D(gray, -1, kernel)

        h, w = residual.shape
        step = 16
        patch_vars: List[float] = []
        for i in range(0, h - step, step):
            for j in range(0, w - step, step):
                patch_vars.append(float(np.var(residual[i:i + step, j:j + step])))

        if not patch_vars:
            return 0.5, {"reason": "image_too_small_for_prnu"}

        mean_var = float(np.mean(patch_vars))
        std_var = float(np.std(patch_vars))

        if mean_var < 1.5:
            art_score = 0.90
        elif mean_var < 6.0:
            art_score = 0.90 - 0.35 * ((mean_var - 1.5) / 4.5)
        elif mean_var < 20.0:
            art_score = 0.55 - 0.30 * ((mean_var - 6.0) / 14.0)
        elif mean_var < 70.0:
            art_score = 0.25 - 0.20 * ((mean_var - 20.0) / 50.0)
        else:
            art_score = 0.10

        art_score = float(np.clip(art_score, 0.02, 0.98))
        return art_score, {
            "noise_mean_variance": round(mean_var, 3),
            "noise_std_variance": round(std_var, 3),
            "art_score": round(art_score, 4),
        }

    # ------------------------------------------------------------------
    # Face detection (lazy-loaded)
    # ------------------------------------------------------------------

    def _detect_faces(self, image_bgr: np.ndarray) -> list:
        """Detect faces using the project's existing FaceDetector (YuNet)."""
        if self._face_detector is None:
            try:
                from app.video.face_detection.face_detector import FaceDetector
                self._face_detector = FaceDetector(
                    conf_threshold=self._face_conf
                )
            except Exception as exc:
                logger.warning(
                    "ContentClassifier: FaceDetector unavailable (%s); "
                    "using empty face list.",
                    exc,
                )
                return []
        try:
            return self._face_detector.detect(image_bgr)
        except Exception as exc:
            logger.debug(
                "ContentClassifier: face detection raised: %s", exc
            )
            return []

    # ------------------------------------------------------------------
    # Reason builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_art_reason(
        c1: Dict[str, Any],
        c2: Dict[str, Any],
        c3: Dict[str, Any],
        c4: Dict[str, Any],
    ) -> str:
        """Compose a concise human-readable explanation for a DIGITAL_ART result."""
        signals: List[str] = []

        if c1.get("line_art_index", 0.0) >= 0.35:
            signals.append(
                "high line-art edge density "
                f"(edge_density={c1.get('edge_density', 0.0):.3f}, "
                f"flat_ratio={c1.get('flat_ratio', 0.0):.3f})"
            )
        if c2.get("flatness_score", 0.0) >= 0.45:
            signals.append(
                "flat-shaded quantised colour palette "
                f"(intra_cluster_var="
                f"{c2.get('mean_intra_cluster_variance', 0.0):.1f})"
            )
        if c3.get("art_score", 0.0) >= 0.30:
            signals.append(
                "face proportions deviate from natural human anthropometric ratios"
            )
        if c4.get("art_score", 0.0) >= 0.60:
            signals.append(
                "near-absent PRNU sensor noise "
                f"(noise_var={c4.get('noise_mean_variance', 0.0):.2f})"
            )

        if signals:
            return (
                "Image classified as non-photographic digital art based on: "
                + "; ".join(signals)
                + ". Deepfake biometric forensics are not applicable to this "
                "content type."
            )
        return (
            "Aggregate multi-cue signal indicates non-photographic digital "
            "artwork or wallpaper. Deepfake biometric forensics are not "
            "applicable to this content type."
        )
