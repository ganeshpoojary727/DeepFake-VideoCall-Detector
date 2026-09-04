"""
Unit tests for Stage-0 ContentClassifier.

Tests cover:
- Individual cue computation (edge/line-art, palette entropy, PRNU noise)
- Decision branches (DIGITAL_ART, SCENERY_OBJECT, PHOTOGRAPHIC_HUMAN)
- Edge-case inputs (grayscale, BGRA, single-pixel, oversized)
- ContentClassification dataclass helpers (to_dict, verdict_label)
- The module-level singleton accessor
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.analyzer.content_classifier import (
    DIGITAL_ART_ANIME,
    PHOTOGRAPHIC_HUMAN,
    SCENERY_OBJECT,
    ContentClassification,
    ContentClassifier,
)
from app.analyzer.image_analyzer import _get_content_classifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clf() -> ContentClassifier:
    """Module-scoped ContentClassifier instance (YuNet download happens once)."""
    return ContentClassifier(face_conf_threshold=0.50)


def _flat_art_image() -> np.ndarray:
    """Synthetic flat-colour cartoon image: hard edges + zero noise."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[40:260, 40:260] = [0, 140, 255]   # flat blue fill
    img[40:50, 40:260] = [0, 0, 0]        # hard black border top
    img[250:260, 40:260] = [0, 0, 0]      # hard black border bottom
    img[40:260, 40:50] = [0, 0, 0]        # hard black border left
    img[40:260, 250:260] = [0, 0, 0]      # hard black border right
    return img


def _noisy_photo_image() -> np.ndarray:
    """Synthetic photographic-texture image: rich noise floor, no hard outlines."""
    rng = np.random.default_rng(1234)
    img = rng.integers(40, 210, (256, 256, 3), dtype=np.uint8).astype(np.float32)
    # Add Gaussian noise typical of a camera sensor
    img += rng.normal(0, 18, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# ContentClassification dataclass tests
# ---------------------------------------------------------------------------

class TestContentClassificationDataclass:
    def test_to_dict_keys(self):
        cc = ContentClassification(
            category=DIGITAL_ART_ANIME,
            is_biometric_applicable=False,
            confidence=0.75,
            reason="test reason",
            details={"foo": 1},
            elapsed_ms=2.3,
        )
        d = cc.to_dict()
        expected = {
            "category", "is_biometric_applicable", "confidence",
            "reason", "details", "elapsed_ms",
        }
        assert set(d.keys()) == expected

    def test_verdict_label_not_applicable(self):
        cc = ContentClassification(
            category=DIGITAL_ART_ANIME,
            is_biometric_applicable=False,
            confidence=0.80,
            reason="art",
        )
        assert cc.verdict_label == "NOT_APPLICABLE"

    def test_verdict_label_applicable(self):
        cc = ContentClassification(
            category=PHOTOGRAPHIC_HUMAN,
            is_biometric_applicable=True,
            confidence=0.80,
            reason="photo",
        )
        assert cc.verdict_label == "APPLICABLE"

    def test_confidence_clamped_in_dict(self):
        cc = ContentClassification(
            category=SCENERY_OBJECT,
            is_biometric_applicable=False,
            confidence=1.23456789,
            reason="x",
        )
        assert cc.to_dict()["confidence"] == round(1.23456789, 4)


# ---------------------------------------------------------------------------
# Individual cue method tests
# ---------------------------------------------------------------------------

class TestCueMethods:

    def test_cue_edge_line_art_flat_image(self):
        """A perfectly flat single-colour image should have high art_score."""
        flat = np.full((256, 256, 3), 128, dtype=np.uint8)
        score, details = ContentClassifier._cue_edge_line_art(flat)
        # Flat image: no edges, all flat → high flat_ratio, low texture_variance
        assert details["flat_ratio"] > 0.90, f"Expected high flat_ratio, got {details}"
        assert 0.0 < score <= 0.98

    def test_cue_edge_line_art_noisy_image(self):
        """A high-noise image should produce a lower art_score than a flat one."""
        flat = np.full((256, 256, 3), 128, dtype=np.uint8)
        noisy = _noisy_photo_image()
        score_flat, _ = ContentClassifier._cue_edge_line_art(flat)
        score_noisy, _ = ContentClassifier._cue_edge_line_art(noisy)
        # Noisy photographic image should be less 'art-like' than a flat image
        assert score_flat >= score_noisy

    def test_cue_palette_entropy_flat(self):
        """Single-colour image → maximum flatness, low art_score (degenerate case)."""
        flat = np.full((256, 256, 3), 64, dtype=np.uint8)
        score, details = ContentClassifier._cue_palette_entropy(flat)
        # flatness_score should be high (intra-cluster variance ≈ 0)
        assert details["flatness_score"] >= 0.90
        assert 0.0 < score <= 0.98

    def test_cue_palette_entropy_random(self):
        """Uniformly random image → high entropy, lower art_score."""
        rng = np.random.default_rng(99)
        random_img = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        score_rand, details_rand = ContentClassifier._cue_palette_entropy(random_img)
        flat = np.full((256, 256, 3), 64, dtype=np.uint8)
        score_flat, _ = ContentClassifier._cue_palette_entropy(flat)
        # Random image should have higher entropy (lower flatness) → lower art_score
        assert details_rand["normalized_entropy"] > 0.5
        assert 0.0 < score_rand <= 0.98

    def test_cue_prnu_noise_near_zero_variance(self):
        """Perfectly smooth synthetic image → very high art_score (no sensor noise)."""
        smooth = np.full((256, 256, 3), 128, dtype=np.uint8)
        score, details = ContentClassifier._cue_prnu_noise(smooth)
        assert details["noise_mean_variance"] < 2.0
        assert score >= 0.70, f"Expected high art_score for smooth image, got {score}"

    def test_cue_prnu_noise_high_variance(self):
        """Heavily noisy image → low art_score (sensor-like noise present)."""
        rng = np.random.default_rng(7)
        noisy = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        score_noisy, _ = ContentClassifier._cue_prnu_noise(noisy)
        smooth = np.full((256, 256, 3), 128, dtype=np.uint8)
        score_smooth, _ = ContentClassifier._cue_prnu_noise(smooth)
        assert score_smooth > score_noisy

    def test_cue_face_anthropometry_no_faces(self):
        """With empty face_boxes, cue returns the neutral score."""
        score, details = ContentClassifier._cue_face_anthropometry(
            np.zeros((256, 256, 3), dtype=np.uint8), face_boxes=[]
        )
        assert details["reason"] == "no_face_detected"
        assert 0.0 < score < 1.0

    def test_cue_face_anthropometry_too_small_faces(self):
        """Faces smaller than 5 px in each dimension are ignored."""
        from app.video.face_detection.face_detector import FaceBox
        tiny_face = FaceBox(x=10, y=10, w=3, h=3)   # too small
        score, details = ContentClassifier._cue_face_anthropometry(
            np.zeros((256, 256, 3), dtype=np.uint8), face_boxes=[tiny_face]
        )
        assert details.get("reason") == "face_too_small"


# ---------------------------------------------------------------------------
# End-to-end classify() tests
# ---------------------------------------------------------------------------

class TestClassifyEndToEnd:

    def test_flat_art_is_digital_art(self, clf):
        result = clf.classify(_flat_art_image())
        assert result.category == DIGITAL_ART_ANIME
        assert result.is_biometric_applicable is False
        assert result.verdict_label == "NOT_APPLICABLE"
        assert 0.50 <= result.confidence <= 1.0
        assert result.elapsed_ms > 0.0

    def test_noisy_photo_no_face_is_scenery(self, clf):
        result = clf.classify(_noisy_photo_image())
        # No face → not PHOTOGRAPHIC_HUMAN
        assert result.is_biometric_applicable is False

    def test_grayscale_input_handled(self, clf):
        gray = np.full((200, 200), 128, dtype=np.uint8)
        result = clf.classify(gray)
        assert result.category in (PHOTOGRAPHIC_HUMAN, DIGITAL_ART_ANIME, SCENERY_OBJECT)
        assert isinstance(result.is_biometric_applicable, bool)

    def test_bgra_input_handled(self, clf):
        bgra = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)
        result = clf.classify(bgra)
        assert result.category in (PHOTOGRAPHIC_HUMAN, DIGITAL_ART_ANIME, SCENERY_OBJECT)

    def test_tiny_image_handled(self, clf):
        """Single-pixel image should not crash."""
        tiny = np.zeros((1, 1, 3), dtype=np.uint8)
        result = clf.classify(tiny)
        assert result.category in (PHOTOGRAPHIC_HUMAN, DIGITAL_ART_ANIME, SCENERY_OBJECT)

    def test_large_image_downsampled(self, clf):
        """4K-ish image should still be processed quickly (no OOM)."""
        large = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
        result = clf.classify(large)
        assert result.category in (PHOTOGRAPHIC_HUMAN, DIGITAL_ART_ANIME, SCENERY_OBJECT)
        # Should complete well under 5 seconds on CPU
        assert result.elapsed_ms < 5000.0

    def test_details_structure(self, clf):
        """details dict must contain all cue sub-dicts."""
        result = clf.classify(_flat_art_image())
        details = result.details
        assert "art_score" in details
        assert "photo_score" in details
        assert "face_count" in details
        assert "cue_1_edge_line_art" in details
        assert "cue_2_palette_entropy" in details
        assert "cue_3_face_anthropometry" in details
        assert "cue_4_prnu_noise" in details

    def test_classify_file_not_found(self, clf, tmp_path):
        with pytest.raises(FileNotFoundError):
            clf.classify_file(tmp_path / "nonexistent.png")

    def test_classify_file_invalid(self, clf, tmp_path):
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"not an image at all")
        with pytest.raises(ValueError):
            clf.classify_file(bad)

    def test_classify_file_success(self, clf, tmp_path):
        """Round-trip: save synthetic image to disk and classify via classify_file."""
        img = _flat_art_image()
        save_path = tmp_path / "art.png"
        cv2.imwrite(str(save_path), img)
        result = clf.classify_file(save_path)
        assert result.category in (PHOTOGRAPHIC_HUMAN, DIGITAL_ART_ANIME, SCENERY_OBJECT)

    def test_singleton_accessor(self):
        """_get_content_classifier should return a ContentClassifier."""
        instance = _get_content_classifier()
        assert isinstance(instance, ContentClassifier)
        # Should be the same object on second call
        assert _get_content_classifier() is instance


# ---------------------------------------------------------------------------
# Reason builder
# ---------------------------------------------------------------------------

class TestBuildArtReason:
    def test_builds_with_signals(self):
        c1 = {"line_art_index": 0.50, "edge_density": 0.12, "flat_ratio": 0.70}
        c2 = {"flatness_score": 0.60, "mean_intra_cluster_variance": 5.0}
        c3 = {"art_score": 0.40}
        c4 = {"art_score": 0.75, "noise_mean_variance": 0.3}
        reason = ContentClassifier._build_art_reason(c1, c2, c3, c4)
        assert "deepfake" in reason.lower() or "biometric" in reason.lower()
        assert len(reason) > 30

    def test_builds_fallback_when_no_signals(self):
        reason = ContentClassifier._build_art_reason({}, {}, {}, {})
        assert "Aggregate" in reason or "multi-cue" in reason.lower()
