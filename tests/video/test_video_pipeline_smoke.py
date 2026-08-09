"""Phase 1 Video Pipeline Foundation -- End-to-End Smoke Test with Real Videos.

This test verifies the COMPLETE video preprocessing pipeline using actual
dataset videos (FaceForensics++ and Celeb-DF v2), confirming:
  1. Dataset discovery (files exist on disk)
  2. Video decoding (cv2 produces non-black RGB frames)
  3. Frame sampling (uniform 16-frame extraction)
  4. Face detection (YuNet detects faces in real video frames)
  5. Face cropping (224x224 face crops)
  6. Sequence building ([T, H, W, C] packing)
  7. Tensor conversion ([T, C, H, W] float32)
  8. ImageNet normalization (mean/std applied, no NaN/Inf)
  9. Batch collation (VideoSample -> collated dict)
 10. CPU/GPU transfer roundtrip

NO model instantiation, NO training, NO audio changes.
"""

from __future__ import annotations

import os
import glob
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest
import torch

# -- Project root ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # tests/video/ -> project root

# -- Dataset paths -----------------------------------------------------------
FFPP_ORIGINAL_DIR = PROJECT_ROOT / "datasets" / "video" / "faceforensics" / "original"
FFPP_DEEPFAKES_DIR = PROJECT_ROOT / "datasets" / "video" / "faceforensics" / "Deepfakes"
CELEBDF_REAL_DIR = PROJECT_ROOT / "datasets" / "video" / "celebdfv2" / "Celeb-real"
CELEBDF_SYNTH_DIR = PROJECT_ROOT / "datasets" / "video" / "celebdfv2" / "Celeb-synthesis"

# -- Constants ---------------------------------------------------------------
NUM_FRAMES = 16
FACE_SIZE = (224, 224)  # (W, H)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _list_mp4s(directory: Path, limit: int = 3) -> List[Path]:
    """Return up to *limit* .mp4 file paths from *directory*."""
    files = sorted(directory.glob("*.mp4"))
    return files[:limit]


def _assert_finite(tensor: torch.Tensor, tag: str) -> None:
    """Assert that a tensor has NO NaN or Inf values."""
    assert torch.isfinite(tensor).all(), (
        f"[{tag}] Tensor contains NaN or Inf! "
        f"NaN count: {torch.isnan(tensor).sum().item()}, "
        f"Inf count: {torch.isinf(tensor).sum().item()}, "
        f"shape: {tensor.shape}"
    )


def _is_non_black(frame: np.ndarray, threshold: float = 5.0) -> bool:
    """Return True if the frame is NOT entirely black (mean pixel > threshold)."""
    return float(frame.mean()) > threshold


# ------------------------------------------------------------------------------
# STAGE 1: Dataset Discovery
# ------------------------------------------------------------------------------

class TestStage1DatasetDiscovery:
    """Verify that datasets exist on disk with real video files."""

    def test_ffpp_original_directory_exists(self) -> None:
        assert FFPP_ORIGINAL_DIR.is_dir(), (
            f"FF++ original directory missing: {FFPP_ORIGINAL_DIR}"
        )

    def test_ffpp_original_has_videos(self) -> None:
        mp4s = list(FFPP_ORIGINAL_DIR.glob("*.mp4"))
        assert len(mp4s) >= 100, (
            f"Expected >=100 FF++ original videos, found {len(mp4s)}"
        )
        print(f"  [DISCOVERY] FF++ original: {len(mp4s)} videos")

    def test_ffpp_deepfakes_directory_exists(self) -> None:
        assert FFPP_DEEPFAKES_DIR.is_dir(), (
            f"FF++ Deepfakes directory missing: {FFPP_DEEPFAKES_DIR}"
        )

    def test_ffpp_deepfakes_has_videos(self) -> None:
        mp4s = list(FFPP_DEEPFAKES_DIR.glob("*.mp4"))
        assert len(mp4s) >= 100, (
            f"Expected >=100 FF++ Deepfakes videos, found {len(mp4s)}"
        )
        print(f"  [DISCOVERY] FF++ Deepfakes: {len(mp4s)} videos")

    def test_celebdf_real_directory_exists(self) -> None:
        assert CELEBDF_REAL_DIR.is_dir(), (
            f"CelebDF real directory missing: {CELEBDF_REAL_DIR}"
        )

    def test_celebdf_real_has_videos(self) -> None:
        mp4s = list(CELEBDF_REAL_DIR.glob("*.mp4"))
        assert len(mp4s) >= 50, (
            f"Expected >=50 CelebDF real videos, found {len(mp4s)}"
        )
        print(f"  [DISCOVERY] CelebDF real: {len(mp4s)} videos")

    def test_celebdf_synthesis_directory_exists(self) -> None:
        assert CELEBDF_SYNTH_DIR.is_dir(), (
            f"CelebDF synthesis directory missing: {CELEBDF_SYNTH_DIR}"
        )

    def test_celebdf_synthesis_has_videos(self) -> None:
        mp4s = list(CELEBDF_SYNTH_DIR.glob("*.mp4"))
        assert len(mp4s) >= 50, (
            f"Expected >=50 CelebDF synthesis videos, found {len(mp4s)}"
        )
        print(f"  [DISCOVERY] CelebDF synthesis: {len(mp4s)} videos")


# ------------------------------------------------------------------------------
# STAGE 2: Video Decoding with Real Files
# ------------------------------------------------------------------------------

class TestStage2VideoDecoding:
    """Decode real .mp4 files and verify frame format/content."""

    @pytest.fixture
    def decoder(self):
        from app.video.preprocessing.video_decoder import VideoDecoder
        return VideoDecoder()

    @pytest.fixture
    def ffpp_videos(self) -> List[Path]:
        return _list_mp4s(FFPP_ORIGINAL_DIR, limit=3)

    @pytest.fixture
    def celebdf_videos(self) -> List[Path]:
        return _list_mp4s(CELEBDF_REAL_DIR, limit=2)

    def test_ffpp_decode_produces_frames(self, decoder, ffpp_videos) -> None:
        """FF++ original videos decode to non-empty frame lists."""
        assert len(ffpp_videos) >= 1, "No FF++ videos found for decoding test"

        for vpath in ffpp_videos:
            t0 = time.perf_counter()
            frames = decoder.decode(str(vpath))
            elapsed = time.perf_counter() - t0

            assert len(frames) > 0, f"Decoded 0 frames from {vpath.name}"
            print(
                f"  [DECODE] {vpath.name}: {len(frames)} frames, "
                f"{elapsed:.2f}s, first frame {frames[0].shape}"
            )

    def test_decoded_frames_are_rgb_uint8(self, decoder, ffpp_videos) -> None:
        """Each decoded frame is an RGB uint8 ndarray with shape [H, W, 3]."""
        frames = decoder.decode(str(ffpp_videos[0]))
        for i, frame in enumerate(frames[:5]):  # Check first 5
            assert isinstance(frame, np.ndarray), f"Frame {i} is not ndarray"
            assert frame.ndim == 3, f"Frame {i} ndim={frame.ndim}, expected 3"
            assert frame.shape[2] == 3, f"Frame {i} channels={frame.shape[2]}, expected 3"
            assert frame.dtype == np.uint8, f"Frame {i} dtype={frame.dtype}, expected uint8"

    def test_decoded_frames_are_not_black(self, decoder, ffpp_videos) -> None:
        """Decoded frames contain actual content, not all-black placeholders."""
        frames = decoder.decode(str(ffpp_videos[0]))
        non_black_count = sum(1 for f in frames[:10] if _is_non_black(f))
        assert non_black_count >= 8, (
            f"Only {non_black_count}/10 frames are non-black -- "
            f"decoder may be producing garbage"
        )
        print(f"  [DECODE] Non-black frames: {non_black_count}/10")

    def test_celebdf_decode_produces_frames(self, decoder, celebdf_videos) -> None:
        """CelebDF videos also decode correctly."""
        assert len(celebdf_videos) >= 1, "No CelebDF videos found for decoding test"

        frames = decoder.decode(str(celebdf_videos[0]))
        assert len(frames) > 0, f"Decoded 0 frames from {celebdf_videos[0].name}"
        assert _is_non_black(frames[0]), "First CelebDF frame is black"
        print(
            f"  [DECODE] CelebDF {celebdf_videos[0].name}: "
            f"{len(frames)} frames, shape {frames[0].shape}"
        )


# ------------------------------------------------------------------------------
# STAGE 3: Frame Sampling
# ------------------------------------------------------------------------------

class TestStage3FrameSampling:
    """Sample exactly 16 frames from decoded video using uniform strategy."""

    @pytest.fixture
    def all_frames(self) -> List[np.ndarray]:
        from app.video.preprocessing.video_decoder import VideoDecoder
        vpath = _list_mp4s(FFPP_ORIGINAL_DIR, limit=1)[0]
        return VideoDecoder().decode(str(vpath))

    def test_uniform_sampling_returns_16_frames(self, all_frames) -> None:
        from app.video.preprocessing.frame_sampler import FrameSampler
        sampler = FrameSampler(num_frames=NUM_FRAMES, strategy="uniform")
        sampled = sampler.sample(all_frames)
        assert len(sampled) == NUM_FRAMES, (
            f"Expected {NUM_FRAMES} sampled frames, got {len(sampled)}"
        )
        print(f"  [SAMPLE] Input: {len(all_frames)} -> Output: {len(sampled)} frames")

    def test_sampled_frame_shapes_match_originals(self, all_frames) -> None:
        from app.video.preprocessing.frame_sampler import FrameSampler
        sampler = FrameSampler(num_frames=NUM_FRAMES, strategy="uniform")
        sampled = sampler.sample(all_frames)
        h_orig, w_orig = all_frames[0].shape[:2]
        for i, frame in enumerate(sampled):
            assert frame.shape[:2] == (h_orig, w_orig), (
                f"Frame {i} shape mismatch: {frame.shape[:2]} vs ({h_orig}, {w_orig})"
            )

    def test_sampled_frames_are_non_black(self, all_frames) -> None:
        from app.video.preprocessing.frame_sampler import FrameSampler
        sampler = FrameSampler(num_frames=NUM_FRAMES, strategy="uniform")
        sampled = sampler.sample(all_frames)
        non_black = sum(1 for f in sampled if _is_non_black(f))
        assert non_black == NUM_FRAMES, (
            f"Only {non_black}/{NUM_FRAMES} sampled frames are non-black"
        )


# ------------------------------------------------------------------------------
# STAGE 4: Face Detection on Real Frames
# ------------------------------------------------------------------------------

class TestStage4FaceDetection:
    """Run YuNet face detector on real video frames."""

    @pytest.fixture
    def sampled_frames(self) -> List[np.ndarray]:
        """Decode + sample 16 frames from a real FF++ video."""
        from app.video.preprocessing.video_decoder import VideoDecoder
        from app.video.preprocessing.frame_sampler import FrameSampler
        vpath = _list_mp4s(FFPP_ORIGINAL_DIR, limit=1)[0]
        all_frames = VideoDecoder().decode(str(vpath))
        return FrameSampler(num_frames=NUM_FRAMES, strategy="uniform").sample(all_frames)

    @pytest.fixture
    def detector(self):
        from app.video.face_detection.face_detector import FaceDetector
        return FaceDetector(conf_threshold=0.5)

    def test_face_detection_on_real_frames(self, detector, sampled_frames) -> None:
        """Face detector finds faces in >=60% of sampled frames."""
        from app.video.face_detection.face_detector import FaceBox

        detection_count = 0
        for i, frame in enumerate(sampled_frames):
            # FaceDetector expects BGR, decoded frames are RGB -> convert
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            box = detector.detect_largest(bgr_frame)
            if box is not None:
                detection_count += 1
                if i == 0:
                    print(
                        f"  [DETECT] Frame 0: face at "
                        f"({box.x}, {box.y}, {box.w}, {box.h}), area={box.area}"
                    )

        rate = detection_count / len(sampled_frames)
        print(f"  [DETECT] Detection rate: {detection_count}/{len(sampled_frames)} = {rate:.1%}")
        assert rate >= 0.6, (
            f"Face detection rate {rate:.1%} below 60% threshold"
        )

    def test_face_box_has_positive_dimensions(self, detector, sampled_frames) -> None:
        """Detected face boxes have positive width and height."""
        bgr_frame = cv2.cvtColor(sampled_frames[0], cv2.COLOR_RGB2BGR)
        boxes = detector.detect(bgr_frame)
        for box in boxes:
            assert box.w > 0, f"Face box width <=0: {box.w}"
            assert box.h > 0, f"Face box height <=0: {box.h}"
            assert box.area > 0, f"Face box area <=0: {box.area}"


# ------------------------------------------------------------------------------
# STAGE 5: Face Cropping
# ------------------------------------------------------------------------------

class TestStage5FaceCropping:
    """Crop detected faces to 224x224."""

    @pytest.fixture
    def frame_and_detector(self):
        from app.video.preprocessing.video_decoder import VideoDecoder
        from app.video.preprocessing.frame_sampler import FrameSampler
        from app.video.face_detection.face_detector import FaceDetector

        vpath = _list_mp4s(FFPP_ORIGINAL_DIR, limit=1)[0]
        all_frames = VideoDecoder().decode(str(vpath))
        sampled = FrameSampler(num_frames=NUM_FRAMES, strategy="uniform").sample(all_frames)
        detector = FaceDetector(conf_threshold=0.5)
        return sampled, detector

    def test_crop_face_produces_correct_shape(self, frame_and_detector) -> None:
        """Face crop outputs (224, 224, 3) arrays."""
        sampled, detector = frame_and_detector
        crops = []
        for frame in sampled:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            box = detector.detect_largest(bgr)
            if box is not None:
                crop = detector.crop_face(bgr, box, target_size=FACE_SIZE)
                crops.append(crop)
            else:
                # Center crop fallback for frames without detected faces
                h, w = frame.shape[:2]
                side = min(h, w)
                y_start = (h - side) // 2
                x_start = (w - side) // 2
                crop = frame[y_start:y_start + side, x_start:x_start + side]
                crop = cv2.resize(crop, FACE_SIZE)
                crops.append(crop)

        assert len(crops) == NUM_FRAMES, f"Expected {NUM_FRAMES} crops, got {len(crops)}"
        for i, crop in enumerate(crops):
            assert crop.shape == (FACE_SIZE[1], FACE_SIZE[0], 3), (
                f"Crop {i} shape {crop.shape} != expected (224, 224, 3)"
            )
        print(f"  [CROP] All {len(crops)} face crops are (224, 224, 3) [OK]")

    def test_crops_are_not_black(self, frame_and_detector) -> None:
        """Cropped faces contain actual content."""
        sampled, detector = frame_and_detector
        bgr = cv2.cvtColor(sampled[0], cv2.COLOR_RGB2BGR)
        box = detector.detect_largest(bgr)
        if box is not None:
            crop = detector.crop_face(bgr, box, target_size=FACE_SIZE)
            assert _is_non_black(crop), "Face crop is all black"
            print(f"  [CROP] Mean pixel value: {crop.mean():.1f} (non-black [OK])")
        else:
            pytest.skip("No face detected in first frame")


# ------------------------------------------------------------------------------
# STAGE 6: Sequence Building
# ------------------------------------------------------------------------------

class TestStage6SequenceBuilding:
    """Pack 16 face crops into [T, H, W, C] ndarray."""

    @pytest.fixture
    def face_crops(self) -> List[np.ndarray]:
        """Generate 16 face crops from a real video."""
        from app.video.preprocessing.video_decoder import VideoDecoder
        from app.video.preprocessing.frame_sampler import FrameSampler
        from app.video.face_detection.face_detector import FaceDetector

        vpath = _list_mp4s(FFPP_ORIGINAL_DIR, limit=1)[0]
        all_frames = VideoDecoder().decode(str(vpath))
        sampled = FrameSampler(num_frames=NUM_FRAMES, strategy="uniform").sample(all_frames)
        detector = FaceDetector(conf_threshold=0.5)

        crops = []
        for frame in sampled:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            box = detector.detect_largest(bgr)
            if box is not None:
                crop = detector.crop_face(bgr, box, target_size=FACE_SIZE)
            else:
                h, w = frame.shape[:2]
                side = min(h, w)
                y0 = (h - side) // 2
                x0 = (w - side) // 2
                crop = cv2.resize(frame[y0:y0+side, x0:x0+side], FACE_SIZE)
            crops.append(crop)
        return crops

    def test_sequence_builder_output_shape(self, face_crops) -> None:
        from app.video.preprocessing.sequence_builder import SequenceBuilder
        builder = SequenceBuilder(sequence_length=NUM_FRAMES, pad_if_short=True)
        sequence = builder.build(face_crops)
        assert sequence.shape == (NUM_FRAMES, FACE_SIZE[1], FACE_SIZE[0], 3), (
            f"Sequence shape {sequence.shape} != expected (16, 224, 224, 3)"
        )
        print(f"  [SEQUENCE] Output shape: {sequence.shape} [OK]")

    def test_sequence_builder_preserves_dtype(self, face_crops) -> None:
        from app.video.preprocessing.sequence_builder import SequenceBuilder
        builder = SequenceBuilder(sequence_length=NUM_FRAMES)
        sequence = builder.build(face_crops)
        assert sequence.dtype == np.uint8, f"Sequence dtype {sequence.dtype} != uint8"


# ------------------------------------------------------------------------------
# STAGE 7: Tensor Conversion
# ------------------------------------------------------------------------------

class TestStage7TensorConversion:
    """Convert [T, H, W, C] ndarray -> [T, C, H, W] float32 tensor."""

    @pytest.fixture
    def sequence_array(self) -> np.ndarray:
        """Build a [16, 224, 224, 3] uint8 sequence from real video."""
        from app.video.preprocessing.video_decoder import VideoDecoder
        from app.video.preprocessing.frame_sampler import FrameSampler
        from app.video.preprocessing.sequence_builder import SequenceBuilder
        from app.video.face_detection.face_detector import FaceDetector

        vpath = _list_mp4s(FFPP_ORIGINAL_DIR, limit=1)[0]
        all_frames = VideoDecoder().decode(str(vpath))
        sampled = FrameSampler(num_frames=NUM_FRAMES).sample(all_frames)
        detector = FaceDetector(conf_threshold=0.5)

        crops = []
        for f in sampled:
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            box = detector.detect_largest(bgr)
            if box:
                crops.append(detector.crop_face(bgr, box, target_size=FACE_SIZE))
            else:
                h, w = f.shape[:2]
                s = min(h, w)
                crops.append(cv2.resize(f[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2], FACE_SIZE))

        return SequenceBuilder(NUM_FRAMES).build(crops)

    def test_tensor_shape(self, sequence_array) -> None:
        from app.video.preprocessing.video_tensor_converter import VideoTensorConverter
        converter = VideoTensorConverter(scale_to_unit=True)
        tensor = converter.to_tensor(sequence_array)
        assert tensor.shape == (NUM_FRAMES, 3, FACE_SIZE[1], FACE_SIZE[0]), (
            f"Tensor shape {tensor.shape} != expected (16, 3, 224, 224)"
        )
        print(f"  [TENSOR] Shape: {tensor.shape} [OK]")

    def test_tensor_dtype_is_float32(self, sequence_array) -> None:
        from app.video.preprocessing.video_tensor_converter import VideoTensorConverter
        tensor = VideoTensorConverter().to_tensor(sequence_array)
        assert tensor.dtype == torch.float32, f"dtype {tensor.dtype} != float32"

    def test_tensor_range_zero_to_one(self, sequence_array) -> None:
        from app.video.preprocessing.video_tensor_converter import VideoTensorConverter
        tensor = VideoTensorConverter(scale_to_unit=True).to_tensor(sequence_array)
        assert tensor.min() >= 0.0, f"Tensor min {tensor.min():.4f} < 0"
        assert tensor.max() <= 1.0, f"Tensor max {tensor.max():.4f} > 1"
        print(f"  [TENSOR] Range: [{tensor.min():.4f}, {tensor.max():.4f}] [OK]")

    def test_tensor_is_finite(self, sequence_array) -> None:
        from app.video.preprocessing.video_tensor_converter import VideoTensorConverter
        tensor = VideoTensorConverter().to_tensor(sequence_array)
        _assert_finite(tensor, "tensor_conversion")


# ------------------------------------------------------------------------------
# STAGE 8: ImageNet Normalization
# ------------------------------------------------------------------------------

class TestStage8Normalization:
    """Apply ImageNet normalization and verify value range + no NaN/Inf."""

    @pytest.fixture
    def pre_norm_tensor(self) -> torch.Tensor:
        """Build a [16, 3, 224, 224] float32 [0,1] tensor from real video."""
        from app.video.preprocessing.video_decoder import VideoDecoder
        from app.video.preprocessing.frame_sampler import FrameSampler
        from app.video.preprocessing.sequence_builder import SequenceBuilder
        from app.video.preprocessing.video_tensor_converter import VideoTensorConverter
        from app.video.face_detection.face_detector import FaceDetector

        vpath = _list_mp4s(FFPP_ORIGINAL_DIR, limit=1)[0]
        all_frames = VideoDecoder().decode(str(vpath))
        sampled = FrameSampler(num_frames=NUM_FRAMES).sample(all_frames)
        detector = FaceDetector(conf_threshold=0.5)

        crops = []
        for f in sampled:
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            box = detector.detect_largest(bgr)
            if box:
                crops.append(detector.crop_face(bgr, box, target_size=FACE_SIZE))
            else:
                h, w = f.shape[:2]
                s = min(h, w)
                crops.append(cv2.resize(f[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2], FACE_SIZE))

        seq = SequenceBuilder(NUM_FRAMES).build(crops)
        return VideoTensorConverter().to_tensor(seq)

    def test_normalizer_instantiation(self) -> None:
        """VideoNormalizer can be instantiated (Optional import fix verified)."""
        from app.video.preprocessing.video_normalizer import VideoNormalizer
        normalizer = VideoNormalizer()
        assert normalizer is not None

    def test_normalized_shape_unchanged(self, pre_norm_tensor) -> None:
        from app.video.preprocessing.video_normalizer import VideoNormalizer
        normalizer = VideoNormalizer()
        normalized = normalizer.normalize(pre_norm_tensor)
        assert normalized.shape == pre_norm_tensor.shape, (
            f"Shape changed after normalization: {pre_norm_tensor.shape} -> {normalized.shape}"
        )

    def test_normalized_range(self, pre_norm_tensor) -> None:
        """ImageNet-normalized values fall in approximately [-2.5, 3.0]."""
        from app.video.preprocessing.video_normalizer import VideoNormalizer
        normalized = VideoNormalizer().normalize(pre_norm_tensor)
        assert normalized.min() >= -3.0, f"Normalized min {normalized.min():.4f} < -3.0"
        assert normalized.max() <= 4.0, f"Normalized max {normalized.max():.4f} > 4.0"
        print(
            f"  [NORM] Range: [{normalized.min():.4f}, {normalized.max():.4f}] [OK]"
        )

    def test_normalized_no_nan_inf(self, pre_norm_tensor) -> None:
        """Critical: No NaN or Inf after normalization."""
        from app.video.preprocessing.video_normalizer import VideoNormalizer
        normalized = VideoNormalizer().normalize(pre_norm_tensor)
        _assert_finite(normalized, "normalization")
        print("  [NORM] No NaN/Inf [OK]")


# ------------------------------------------------------------------------------
# STAGE 9: Batch Collation via VideoSample
# ------------------------------------------------------------------------------

class TestStage9BatchCollation:
    """Create VideoSamples and collate them into a training batch."""

    @pytest.fixture
    def two_samples(self) -> list:
        """Build 2 VideoSample objects from 2 different real videos."""
        from app.video.preprocessing.video_decoder import VideoDecoder
        from app.video.preprocessing.frame_sampler import FrameSampler
        from app.video.preprocessing.sequence_builder import SequenceBuilder
        from app.video.preprocessing.video_tensor_converter import VideoTensorConverter
        from app.video.preprocessing.video_normalizer import VideoNormalizer
        from app.video.face_detection.face_detector import FaceDetector
        from app.video.datasets.video_sample import VideoSample

        decoder = VideoDecoder()
        sampler = FrameSampler(num_frames=NUM_FRAMES)
        builder = SequenceBuilder(NUM_FRAMES)
        converter = VideoTensorConverter()
        normalizer = VideoNormalizer()
        detector = FaceDetector(conf_threshold=0.5)

        # Pick one real and one fake video
        real_path = _list_mp4s(FFPP_ORIGINAL_DIR, limit=1)[0]
        fake_path = _list_mp4s(FFPP_DEEPFAKES_DIR, limit=1)[0]

        samples = []
        for vpath, label in [(real_path, 0), (fake_path, 1)]:
            all_frames = decoder.decode(str(vpath))
            sampled = sampler.sample(all_frames)

            crops = []
            for f in sampled:
                bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                box = detector.detect_largest(bgr)
                if box:
                    crops.append(detector.crop_face(bgr, box, target_size=FACE_SIZE))
                else:
                    h, w = f.shape[:2]
                    s = min(h, w)
                    crops.append(
                        cv2.resize(f[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2], FACE_SIZE)
                    )

            seq = builder.build(crops)
            tensor = converter.to_tensor(seq)
            tensor = normalizer.normalize(tensor)
            _assert_finite(tensor, f"sample_{label}")

            sample = VideoSample(
                tensor=tensor,
                label=label,
                filepath=str(vpath),
                sample_id=vpath.stem,
            )
            samples.append(sample)

        return samples

    def test_collate_batch_tensor_shape(self, two_samples) -> None:
        from app.video.datasets.video_sample import video_collate_fn
        batch = video_collate_fn(two_samples)
        assert "tensor" in batch
        assert "label" in batch
        expected_shape = (2, NUM_FRAMES, 3, FACE_SIZE[1], FACE_SIZE[0])
        assert batch["tensor"].shape == expected_shape, (
            f"Batch tensor shape {batch['tensor'].shape} != {expected_shape}"
        )
        print(f"  [COLLATE] Batch tensor shape: {batch['tensor'].shape} [OK]")

    def test_collate_batch_label_shape(self, two_samples) -> None:
        from app.video.datasets.video_sample import video_collate_fn
        batch = video_collate_fn(two_samples)
        assert batch["label"].shape == (2,), (
            f"Label shape {batch['label'].shape} != (2,)"
        )
        assert batch["label"][0].item() == 0, "First sample should be real (label=0)"
        assert batch["label"][1].item() == 1, "Second sample should be fake (label=1)"
        print(f"  [COLLATE] Labels: {batch['label'].tolist()} [OK]")

    def test_collate_batch_no_nan_inf(self, two_samples) -> None:
        from app.video.datasets.video_sample import video_collate_fn
        batch = video_collate_fn(two_samples)
        _assert_finite(batch["tensor"], "collated_batch")
        print("  [COLLATE] No NaN/Inf in batch [OK]")

    def test_collate_batch_dtype(self, two_samples) -> None:
        from app.video.datasets.video_sample import video_collate_fn
        batch = video_collate_fn(two_samples)
        assert batch["tensor"].dtype == torch.float32
        assert batch["label"].dtype == torch.long


# ------------------------------------------------------------------------------
# STAGE 10: CPU / GPU Transfer
# ------------------------------------------------------------------------------

class TestStage10DeviceTransfer:
    """Verify tensor transfer between CPU and GPU preserves shape/dtype/values."""

    @pytest.fixture
    def batch(self) -> dict:
        """Create a collated batch from 2 real videos."""
        from app.video.preprocessing.video_decoder import VideoDecoder
        from app.video.preprocessing.frame_sampler import FrameSampler
        from app.video.preprocessing.sequence_builder import SequenceBuilder
        from app.video.preprocessing.video_tensor_converter import VideoTensorConverter
        from app.video.preprocessing.video_normalizer import VideoNormalizer
        from app.video.face_detection.face_detector import FaceDetector
        from app.video.datasets.video_sample import VideoSample, video_collate_fn

        decoder = VideoDecoder()
        sampler = FrameSampler(num_frames=NUM_FRAMES)
        builder = SequenceBuilder(NUM_FRAMES)
        converter = VideoTensorConverter()
        normalizer = VideoNormalizer()
        detector = FaceDetector(conf_threshold=0.5)

        samples = []
        for vpath, label in zip(
            _list_mp4s(FFPP_ORIGINAL_DIR, 1) + _list_mp4s(FFPP_DEEPFAKES_DIR, 1),
            [0, 1],
        ):
            all_frames = decoder.decode(str(vpath))
            sampled = sampler.sample(all_frames)
            crops = []
            for f in sampled:
                bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                box = detector.detect_largest(bgr)
                if box:
                    crops.append(detector.crop_face(bgr, box, target_size=FACE_SIZE))
                else:
                    h, w = f.shape[:2]
                    s = min(h, w)
                    crops.append(
                        cv2.resize(f[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2], FACE_SIZE)
                    )
            seq = builder.build(crops)
            tensor = normalizer.normalize(converter.to_tensor(seq))
            samples.append(VideoSample(tensor=tensor, label=label,
                                       filepath=str(vpath), sample_id=vpath.stem))

        return video_collate_fn(samples)

    def test_cpu_tensor_shape(self, batch) -> None:
        """Batch tensor on CPU has correct shape."""
        t = batch["tensor"]
        assert t.device.type == "cpu"
        assert t.shape == (2, NUM_FRAMES, 3, FACE_SIZE[1], FACE_SIZE[0])
        print(f"  [DEVICE] CPU shape: {t.shape} [OK]")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU available")
    def test_gpu_transfer_roundtrip(self, batch) -> None:
        """Transfer to GPU and back preserves shape, dtype, and values."""
        cpu_tensor = batch["tensor"]
        cpu_label = batch["label"]

        # CPU -> GPU
        gpu_tensor = cpu_tensor.cuda()
        gpu_label = cpu_label.cuda()
        assert gpu_tensor.device.type == "cuda"
        assert gpu_tensor.shape == cpu_tensor.shape
        assert gpu_tensor.dtype == cpu_tensor.dtype
        _assert_finite(gpu_tensor, "gpu_tensor")
        print(f"  [DEVICE] GPU shape: {gpu_tensor.shape}, device: {gpu_tensor.device} [OK]")

        # GPU -> CPU roundtrip
        back_cpu = gpu_tensor.cpu()
        assert torch.allclose(back_cpu, cpu_tensor, atol=1e-6), (
            "GPU->CPU roundtrip changed tensor values!"
        )
        print("  [DEVICE] GPU->CPU roundtrip: values match [OK]")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU available")
    def test_gpu_memory_footprint(self, batch) -> None:
        """Log GPU memory used by a single batch (sanity check for RTX 4050)."""
        torch.cuda.reset_peak_memory_stats()
        gpu_tensor = batch["tensor"].cuda()
        mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"  [DEVICE] GPU memory for batch: {mem_mb:.1f} MB")
        # A single batch should use <500 MB
        assert mem_mb < 500, f"Single batch uses {mem_mb:.1f} MB -- too much for RTX 4050"
        del gpu_tensor
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------
# BONUS: End-to-End Timing
# ------------------------------------------------------------------------------

class TestEndToEndTiming:
    """Measure total wall-clock time for a single video through the full pipeline."""

    def test_single_video_pipeline_time(self) -> None:
        from app.video.preprocessing.video_decoder import VideoDecoder
        from app.video.preprocessing.frame_sampler import FrameSampler
        from app.video.preprocessing.sequence_builder import SequenceBuilder
        from app.video.preprocessing.video_tensor_converter import VideoTensorConverter
        from app.video.preprocessing.video_normalizer import VideoNormalizer
        from app.video.face_detection.face_detector import FaceDetector

        vpath = _list_mp4s(FFPP_ORIGINAL_DIR, limit=1)[0]

        t0 = time.perf_counter()

        # Decode
        decoder = VideoDecoder()
        all_frames = decoder.decode(str(vpath))
        t_decode = time.perf_counter()

        # Sample
        sampler = FrameSampler(num_frames=NUM_FRAMES)
        sampled = sampler.sample(all_frames)
        t_sample = time.perf_counter()

        # Detect + crop
        detector = FaceDetector(conf_threshold=0.5)
        crops = []
        for f in sampled:
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            box = detector.detect_largest(bgr)
            if box:
                crops.append(detector.crop_face(bgr, box, target_size=FACE_SIZE))
            else:
                h, w = f.shape[:2]
                s = min(h, w)
                crops.append(cv2.resize(f[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2], FACE_SIZE))
        t_detect = time.perf_counter()

        # Sequence -> tensor -> normalize
        seq = SequenceBuilder(NUM_FRAMES).build(crops)
        tensor = VideoTensorConverter().to_tensor(seq)
        normalized = VideoNormalizer().normalize(tensor)
        t_end = time.perf_counter()

        # Validate final tensor
        assert normalized.shape == (NUM_FRAMES, 3, FACE_SIZE[1], FACE_SIZE[0])
        _assert_finite(normalized, "end_to_end")

        total = t_end - t0
        print(
            f"\n  +----------------------------------------------+\n"
            f"  |  END-TO-END PIPELINE TIMING                  |\n"
            f"  +----------------------------------------------+\n"
            f"  |  Video: {vpath.name:<36s} |\n"
            f"  |  Total frames decoded: {len(all_frames):<22d} |\n"
            f"  |  Decode:    {t_decode - t0:>8.3f}s                    |\n"
            f"  |  Sample:    {t_sample - t_decode:>8.3f}s                    |\n"
            f"  |  Detect:    {t_detect - t_sample:>8.3f}s                    |\n"
            f"  |  Tensor:    {t_end - t_detect:>8.3f}s                    |\n"
            f"  |  TOTAL:     {total:>8.3f}s                    |\n"
            f"  |  Output:    {str(normalized.shape):<24s}      |\n"
            f"  |  Finite:    YES [OK]                         |\n"
            f"  +----------------------------------------------+"
        )

        # Single video should process in <30 seconds
        assert total < 30.0, f"Pipeline took {total:.1f}s -- too slow for production"
