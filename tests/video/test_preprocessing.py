"""Unit tests for video preprocessing components."""

import numpy as np
import pytest
import torch
from app.video.exceptions.video_exceptions import PreprocessingError
from app.video.preprocessing.face_aligner import FaceAligner
from app.video.preprocessing.face_cropper import FaceCropper
from app.video.preprocessing.frame_extractor import FrameExtractor
from app.video.preprocessing.frame_sampler import FrameSampler
from app.video.preprocessing.resolution_converter import ResolutionConverter
from app.video.preprocessing.sequence_builder import SequenceBuilder
from app.video.preprocessing.video_normalizer import VideoNormalizer
from app.video.preprocessing.video_tensor_converter import VideoTensorConverter


def test_frame_extractor_from_array(dummy_video_sequence_array):
    extractor = FrameExtractor(max_frames=8)
    frames = extractor.extract(dummy_video_sequence_array)
    assert len(frames) == 8
    assert frames[0].shape == (224, 224, 3)


def test_frame_extractor_single_frame(dummy_frame_array):
    extractor = FrameExtractor()
    frames = extractor.extract(dummy_frame_array)
    assert len(frames) == 1


def test_frame_extractor_invalid_input():
    extractor = FrameExtractor()
    with pytest.raises(PreprocessingError):
        extractor.extract(12345)


def test_face_cropper(dummy_frame_array):
    cropper = FaceCropper(margin=0.2)
    bbox = (20, 20, 100, 100)
    cropped = cropper.crop(dummy_frame_array, bbox=bbox)
    assert cropped.ndim == 3
    assert cropped.shape[2] == 3


def test_face_cropper_no_bbox(dummy_frame_array):
    cropper = FaceCropper()
    cropped = cropper.crop(dummy_frame_array, bbox=None)
    assert cropped.shape == dummy_frame_array.shape


def test_face_cropper_invalid_dim():
    cropper = FaceCropper()
    with pytest.raises(PreprocessingError):
        cropper.crop(np.zeros((100, 100)))


def test_face_aligner(dummy_frame_array):
    aligner = FaceAligner()
    aligned = aligner.align(dummy_frame_array, landmarks=[(10.0, 10.0), (30.0, 10.0)])
    assert aligned.shape == dummy_frame_array.shape


def test_face_aligner_empty_landmarks(dummy_frame_array):
    aligner = FaceAligner()
    aligned = aligner.align(dummy_frame_array, landmarks=[])
    assert aligned.shape == dummy_frame_array.shape


def test_sequence_builder(dummy_frame_array):
    builder = SequenceBuilder(sequence_length=4)
    frames = [dummy_frame_array.copy() for _ in range(2)]
    seq = builder.build(frames)
    assert seq.shape == (4, 224, 224, 3)


def test_sequence_builder_no_padding(dummy_frame_array):
    builder = SequenceBuilder(sequence_length=4, pad_if_short=False)
    frames = [dummy_frame_array.copy() for _ in range(2)]
    seq = builder.build(frames)
    assert seq.shape == (2, 224, 224, 3)


def test_sequence_builder_empty():
    builder = SequenceBuilder()
    with pytest.raises(PreprocessingError):
        builder.build([])


def test_video_normalizer_4d(dummy_video_tensor):
    normalizer = VideoNormalizer()
    norm = normalizer.normalize(dummy_video_tensor)
    assert norm.shape == dummy_video_tensor.shape


def test_video_normalizer_5d(dummy_batch_video_tensor):
    normalizer = VideoNormalizer()
    norm = normalizer.normalize(dummy_batch_video_tensor)
    assert norm.shape == dummy_batch_video_tensor.shape


def test_video_normalizer_invalid_rank():
    normalizer = VideoNormalizer()
    with pytest.raises(PreprocessingError):
        normalizer.normalize(torch.zeros(3, 224, 224))


def test_frame_sampler_uniform(dummy_frame_array):
    frames = [dummy_frame_array for _ in range(30)]
    sampler = FrameSampler(num_frames=10, strategy="uniform")
    sampled = sampler.sample(frames)
    assert len(sampled) == 10


def test_frame_sampler_stride(dummy_frame_array):
    frames = [dummy_frame_array for _ in range(30)]
    sampler = FrameSampler(num_frames=10, strategy="stride", stride=2)
    sampled = sampler.sample(frames)
    assert len(sampled) == 10


def test_frame_sampler_random(dummy_frame_array):
    frames = [dummy_frame_array for _ in range(30)]
    sampler = FrameSampler(num_frames=10, strategy="random")
    sampled = sampler.sample(frames)
    assert len(sampled) == 10


def test_frame_sampler_center(dummy_frame_array):
    frames = [dummy_frame_array for _ in range(30)]
    sampler = FrameSampler(num_frames=10, strategy="center")
    sampled = sampler.sample(frames)
    assert len(sampled) == 10


def test_frame_sampler_empty():
    sampler = FrameSampler()
    with pytest.raises(PreprocessingError):
        sampler.sample([])


def test_frame_sampler_invalid_strategy(dummy_frame_array):
    sampler = FrameSampler(strategy="invalid_strat")
    with pytest.raises(PreprocessingError):
        sampler.sample([dummy_frame_array])


def test_resolution_converter(dummy_frame_array):
    converter = ResolutionConverter(target_resolution=(128, 128))
    resized = converter.convert(dummy_frame_array)
    assert resized.shape == (128, 128, 3)


def test_resolution_converter_same_size(dummy_frame_array):
    converter = ResolutionConverter(target_resolution=(224, 224))
    resized = converter.convert(dummy_frame_array)
    assert resized.shape == (224, 224, 3)


def test_resolution_converter_batch(dummy_frame_array):
    converter = ResolutionConverter(target_resolution=(128, 128))
    batch = converter.convert_batch([dummy_frame_array, dummy_frame_array])
    assert len(batch) == 2
    assert batch[0].shape == (128, 128, 3)


def test_video_tensor_converter(dummy_video_sequence_array):
    converter = VideoTensorConverter()
    tensor = converter.to_tensor(dummy_video_sequence_array)
    assert tensor.shape == (16, 3, 224, 224)
    assert isinstance(tensor, torch.Tensor)


def test_video_tensor_converter_empty_list():
    converter = VideoTensorConverter()
    with pytest.raises(PreprocessingError):
        converter.to_tensor([])
