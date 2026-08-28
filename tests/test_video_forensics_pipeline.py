"""
Comprehensive Verification Suite for Phase 2: Visual & Classical Forensics Subsystem,
Sampling Pipelines, Grad-CAM Saliency, and Structured Visual Telemetry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
import pytest
import torch

from app.analyzer.image_analyzer import ImageAnalyzer
from app.analyzer.image_forensics import ImageForensics
from app.analyzer.video_analyzer import VideoAnalyzer
from app.video.core.base_dataset import BaseVideoDataset
from app.video.datasets.celeb_df_dataset import CelebDFDataset
from app.video.datasets.faceforensics_dataset import FaceForensicsDataset
from app.video.datasets.video_dataset import VideoDataset
from app.video.inference.video_detector import VideoDetector
from app.video.models.efficientnet.model import EfficientNetB4Model
from app.video.preprocessing.face_cropper import FaceCropper
from app.video.preprocessing.frame_sampler import FrameSampler
from app.video.utils.visualization import GradCAM


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_video_frames() -> list[np.ndarray]:
    """Create a list of 30 mock video frames (300x300 BGR) with a synthetic face rectangle."""
    frames = []
    for i in range(30):
        # Background gradient
        frame = np.full((300, 300, 3), fill_value=int(100 + i * 2), dtype=np.uint8)
        # Draw synthetic face-like circle/box in center
        cv2.circle(frame, (150, 150), 50, (200, 180, 170), -1)
        cv2.rectangle(frame, (130, 130), (170, 170), (50, 50, 200), 2)
        frames.append(frame)
    return frames


@pytest.fixture
def temp_visual_manifests(tmp_path: Path) -> Path:
    """Create temporary manifest files for JSON, CSV, and Celeb-DF formats."""
    # 1. JSON manifest
    json_path = tmp_path / "manifest.json"
    json_data = [
        {"filepath": "videos/fake_01.mp4", "label": 1, "sample_id": "fake_01"},
        {"filepath": "videos/real_01.mp4", "label": 0, "sample_id": "real_01"},
    ]
    json_path.write_text(json.dumps(json_data), encoding="utf-8")

    # 2. CSV manifest
    csv_path = tmp_path / "manifest.csv"
    csv_path.write_text(
        "filepath,label,sample_id\n"
        "videos/fake_02.mp4,1,fake_02\n"
        "videos/real_02.mp4,0,real_02\n",
        encoding="utf-8",
    )

    # 3. Celeb-DF TXT manifest
    txt_path = tmp_path / "List_of_testing_videos.txt"
    txt_path.write_text(
        "1 Celeb-synthesis/id0_id1_0000.mp4\n"
        "0 YouTube-real/0000.mp4\n",
        encoding="utf-8",
    )

    return tmp_path


# ── 1. Dataset & Frame Sampling Tests ──────────────────────────────────────────

class TestDatasetAndSamplingPipeline:
    """Test suite for FrameSampler, FaceCropper, and BaseVideoDataset manifest parsing."""

    def test_frame_sampler_uniform_and_stride(self, mock_video_frames: list[np.ndarray]) -> None:
        # Uniform sampling (16 frames)
        sampler_16 = FrameSampler(num_frames=16, strategy="uniform")
        sampled_16 = sampler_16.sample(mock_video_frames)
        assert len(sampled_16) == 16

        # Uniform sampling (32 frames - requires padding/repeats)
        sampler_32 = FrameSampler(num_frames=32, strategy="uniform")
        sampled_32 = sampler_32.sample(mock_video_frames)
        assert len(sampled_32) == 32

        # Fixed stride sampling
        sampler_stride = FrameSampler(num_frames=10, strategy="stride", stride=2)
        sampled_stride = sampler_stride.sample(mock_video_frames)
        assert len(sampled_stride) == 10

        # Sample with metadata
        meta = sampler_16.sample_with_metadata(mock_video_frames, fps=30.0)
        assert len(meta) == 16
        for frame, f_idx, t_sec in meta:
            assert isinstance(frame, np.ndarray)
            assert 0 <= f_idx < 30
            assert t_sec >= 0.0

    def test_face_cropper_margin_and_target_size(self) -> None:
        cropper = FaceCropper(margin=0.2, target_size=(224, 224))
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)

        # Crop with bbox (x1, y1, x2, y2)
        crop, actual_box = cropper.crop_with_bbox_metadata(frame, bbox=(100, 100, 200, 200))
        assert crop.shape == (224, 224, 3)
        assert actual_box[0] <= 100
        assert actual_box[1] <= 100
        assert actual_box[2] >= 200
        assert actual_box[3] >= 200

        # Center-crop fallback
        center_crop, c_box = cropper.crop_with_bbox_metadata(frame, bbox=None)
        assert center_crop.shape == (224, 224, 3)
        assert c_box == (0, 0, 400, 400)

    def test_manifest_parsers_json_csv_txt(self, temp_visual_manifests: Path) -> None:
        # JSON
        rec_json = BaseVideoDataset.parse_manifest_file(temp_visual_manifests / "manifest.json")
        assert len(rec_json) == 2
        assert rec_json[0]["label"] == 1
        assert rec_json[1]["label"] == 0

        # CSV
        rec_csv = BaseVideoDataset.parse_manifest_file(temp_visual_manifests / "manifest.csv")
        assert len(rec_csv) == 2
        assert rec_csv[0]["label"] == 1

        # TXT (Celeb-DF)
        rec_txt = BaseVideoDataset.parse_manifest_file(temp_visual_manifests / "List_of_testing_videos.txt")
        assert len(rec_txt) == 2
        assert rec_txt[0]["label"] == 1
        assert rec_txt[1]["label"] == 0

    def test_specialized_datasets(self, temp_visual_manifests: Path) -> None:
        # FaceForensicsDataset
        ds_ff = FaceForensicsDataset(
            manifest_file=temp_visual_manifests / "manifest.json",
            sequence_length=16,
            target_resolution=(224, 224),
        )
        assert len(ds_ff) == 2
        dist = ds_ff.get_label_distribution()
        assert dist[0] == 1
        assert dist[1] == 1

        # CelebDFDataset
        ds_cdf = CelebDFDataset(
            manifest_file=temp_visual_manifests / "List_of_testing_videos.txt",
            sequence_length=16,
        )
        assert len(ds_cdf) == 2


# ── 2. Classical Forensic Feature Tests ────────────────────────────────────────

class TestClassicalForensicFeatures:
    """Test suite for Error Level Analysis, 2D FFT azimuthal decay, and boundary inconsistency."""

    def test_error_level_analysis(self) -> None:
        image = np.full((256, 256, 3), fill_value=128, dtype=np.uint8)
        # Add high-contrast patch
        cv2.circle(image, (128, 128), 40, (255, 0, 0), -1)

        ela_score, ela_map, details = ImageForensics.compute_ela(image, quality=90)
        assert isinstance(ela_score, float)
        assert 0.0 <= ela_score <= 1.0
        assert ela_map.shape == (256, 256, 3)
        assert "ela_mean" in details
        assert "inconsistency_ratio" in details

    def test_2d_fft_spectrum(self) -> None:
        # 1. Natural smooth image
        smooth = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.circle(smooth, (128, 128), 60, (200, 200, 200), -1)
        smooth = cv2.GaussianBlur(smooth, (15, 15), 0)

        fft_score_smooth, _ = ImageForensics.compute_fft_spectrum(smooth)
        assert 0.0 <= fft_score_smooth <= 1.0

        # 2. Synthetic checkerboard / periodic grid artifact
        grid = np.zeros((256, 256, 3), dtype=np.uint8)
        grid[::4, :] = 255
        grid[:, ::4] = 255
        fft_score_grid, details = ImageForensics.compute_fft_spectrum(grid)
        assert fft_score_grid > fft_score_smooth
        assert "peak_to_mean_ratio" in details

    def test_boundary_inconsistency(self) -> None:
        image = np.full((300, 300, 3), fill_value=100, dtype=np.uint8)
        # Synthetic pasted face with boundary step
        image[80:220, 80:220] = 220

        boundary_score, details = ImageForensics.compute_boundary_inconsistency(image, bbox=(80, 80, 220, 220))
        assert isinstance(boundary_score, float)
        assert 0.0 <= boundary_score <= 1.0
        assert "boundary_step_ratio" in details

    def test_extract_visual_cues_schema(self) -> None:
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        cues = ImageForensics.extract_visual_cues(image)

        assert "ela_discrepancy_score" in cues
        assert "fft_spectral_anomaly" in cues
        assert "boundary_inconsistency" in cues
        assert "combined_score" in cues
        assert 0.0 <= cues["ela_discrepancy_score"] <= 1.0
        assert 0.0 <= cues["fft_spectral_anomaly"] <= 1.0
        assert 0.0 <= cues["boundary_inconsistency"] <= 1.0


# ── 3. Grad-CAM Saliency Tests ─────────────────────────────────────────────────

class TestGradCAMSaliency:
    """Test suite for Grad-CAM generation and heatmap overlay."""

    def test_gradcam_heatmap_generation(self) -> None:
        model = EfficientNetB4Model()
        model.eval()

        gradcam = GradCAM(model)
        dummy_tensor = torch.randn(1, 1, 3, 224, 224, requires_grad=True)

        heatmap = gradcam.generate_heatmap(dummy_tensor, class_idx=1, target_size=(224, 224))
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape == (224, 224)
        assert np.min(heatmap) >= 0.0
        assert np.max(heatmap) <= 1.0 + 1e-5

        # Extract peak coordinates
        px, py = GradCAM.extract_peak_saliency(heatmap)
        assert 0 <= px < 224
        assert 0 <= py < 224

        # Overlay on image
        base_img = np.zeros((224, 224, 3), dtype=np.uint8)
        overlay = GradCAM.overlay_heatmap(base_img, heatmap)
        assert overlay.shape == (224, 224, 3)

        gradcam.remove_hooks()


# ── 4. Structured Output & Diagnostic Telemetry Tests ──────────────────────────

class TestStructuredVisualTelemetry:
    """Test suite asserting strict adherence to the structured output telemetry schema."""

    def test_video_detector_structured_telemetry(self, mock_video_frames: list[np.ndarray]) -> None:
        detector = VideoDetector(sequence_length=16)

        result = detector.predict_detailed(mock_video_frames, fps=30.0)

        # 1. Top level keys
        assert "verdict" in result
        assert "confidence" in result
        assert "raw_scores" in result
        assert "visual_cues" in result
        assert "timeline" in result
        assert "key_artifacts" in result

        # 2. Types & Ranges
        assert result["verdict"] in ("REAL", "FAKE")
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

        # 3. raw_scores
        raw = result["raw_scores"]
        assert "real_prob" in raw
        assert "fake_prob" in raw
        assert 0.0 <= raw["real_prob"] <= 1.0
        assert 0.0 <= raw["fake_prob"] <= 1.0
        assert pytest.approx(raw["real_prob"] + raw["fake_prob"], abs=1e-3) == 1.0

        # 4. visual_cues
        cues = result["visual_cues"]
        assert "ela_discrepancy_score" in cues
        assert "fft_spectral_anomaly" in cues
        assert "boundary_inconsistency" in cues
        assert 0.0 <= cues["ela_discrepancy_score"] <= 1.0
        assert 0.0 <= cues["fft_spectral_anomaly"] <= 1.0
        assert 0.0 <= cues["boundary_inconsistency"] <= 1.0

        # 5. timeline
        timeline = result["timeline"]
        assert isinstance(timeline, list)
        assert len(timeline) == 16
        for item in timeline:
            assert "frame_idx" in item
            assert "timestamp_sec" in item
            assert "spoof_prob" in item
            assert "is_anomaly" in item
            assert isinstance(item["is_anomaly"], bool)
            assert 0.0 <= item["spoof_prob"] <= 1.0

        # 6. key_artifacts
        artifacts = result["key_artifacts"]
        assert isinstance(artifacts, list)
        assert len(artifacts) >= 1
        for art in artifacts:
            assert "frame_idx" in art
            assert "timestamp_sec" in art
            assert "bbox" in art
            assert len(art["bbox"]) == 4
            assert "spoof_prob" in art
            assert "saliency_peak" in art
            assert len(art["saliency_peak"]) == 2

    def test_image_analyzer_structured_telemetry(self, tmp_path: Path) -> None:
        img_path = tmp_path / "test_image.jpg"
        test_img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(test_img, (150, 150), 60, (200, 180, 160), -1)
        cv2.imwrite(str(img_path), test_img)

        analyzer = ImageAnalyzer()
        res = analyzer.analyze_structured(img_path)

        assert res["verdict"] in ("REAL", "FAKE")
        assert "raw_scores" in res
        assert "visual_cues" in res
        assert "timeline" in res
        assert "key_artifacts" in res

        # Also test AnalysisReport
        report = analyzer.analyze(img_path)
        assert report.verdict in ("REAL", "FAKE", "UNCERTAIN")
        assert "visual_cues" in report.metadata
        assert "key_artifacts" in report.metadata
