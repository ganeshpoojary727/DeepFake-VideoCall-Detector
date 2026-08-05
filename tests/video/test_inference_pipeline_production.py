"""Production unit tests for video deepfake inference pipeline module."""

import os
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn

from app.video.configs.inference_config import VideoInferenceConfig
from app.video.exceptions.video_exceptions import ConfigurationError, PreprocessingError
from app.video.models.video_factory import VideoFactory
from app.video.pipeline.inference_pipeline import InferencePipeline, InferenceResult


class DummyVideoModel(nn.Module):
    """Dummy neural network model returning fixed logits [B, 2]."""

    def __init__(self, fake_logit: float = 2.0) -> None:
        super().__init__()
        self.fake_logit = fake_logit
        self.fc = nn.Linear(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W]
        batch_size = x.shape[0]
        # Return logits where fake class index 1 has higher score
        logits = torch.zeros((batch_size, 2), device=x.device)
        logits[:, 0] = 0.0
        logits[:, 1] = self.fake_logit
        return logits


def test_inference_result_dataclass():
    res = InferenceResult(
        is_fake=True,
        is_deepfake=True,
        fake_probability=0.88,
        real_probability=0.12,
        confidence=0.76,
        label=1,
        label_name="fake",
        num_frames=16,
        num_faces_detected=16,
        preprocessing_time_ms=12.5,
        inference_time_ms=5.2,
        postprocessing_time_ms=0.8,
        total_runtime_ms=18.5,
        device="cpu",
        metadata={"test": 123},
    )

    # Test dataclass attributes
    assert res.is_fake is True
    assert res.fake_probability == 0.88

    # Test dict-like subscripting
    assert res["is_fake"] is True
    assert res["fake_probability"] == 0.88
    assert res["test"] == 123
    assert "is_fake" in res
    assert "test" in res
    assert res.get("confidence") == 0.76
    assert res.get("missing", "default") == "default"

    # Test dict serialization
    d = res.to_dict()
    assert isinstance(d, dict)
    assert d["label_name"] == "fake"
    assert "is_fake" in res.keys()
    assert 0.88 in res.values()
    assert ("label_name", "fake") in res.items()

    with pytest.raises(KeyError):
        _ = res["non_existent_key"]


def test_inference_pipeline_invalid_config():
    with pytest.raises(ConfigurationError):
        config = VideoInferenceConfig(sequence_length=-1)
        InferencePipeline(model=DummyVideoModel(), config=config)


def test_inference_pipeline_device_fallback():
    # Force requesting cuda even if not available or fallback to cpu
    config = VideoInferenceConfig(device="cuda:9999")
    pipeline = InferencePipeline(model=DummyVideoModel(), config=config)
    assert pipeline.device.type in ("cuda", "cpu")


def test_inference_pipeline_input_validation():
    pipeline = InferencePipeline(model=DummyVideoModel())

    with pytest.raises(PreprocessingError, match="cannot be None"):
        pipeline.predict_video(None)

    with pytest.raises(PreprocessingError, match="string is empty"):
        pipeline.predict_video("   ")

    with pytest.raises(PreprocessingError, match="does not exist"):
        pipeline.predict_video("non_existent_video_file_12345.mp4")

    with pytest.raises(PreprocessingError, match="bytes buffer is empty"):
        pipeline.predict_video(b"")

    with pytest.raises(PreprocessingError, match="numpy array is empty"):
        pipeline.predict_video(np.array([]))

    with pytest.raises(PreprocessingError, match="frame list is empty"):
        pipeline.predict_video([])

    with pytest.raises(PreprocessingError, match="Unsupported video input type"):
        pipeline.predict_video(12345)


def test_inference_pipeline_single_array_predict(dummy_video_sequence_array):
    model = DummyVideoModel(fake_logit=3.0)
    config = VideoInferenceConfig(sequence_length=8, target_resolution=(112, 112))
    pipeline = InferencePipeline(model=model, config=config)

    res = pipeline.predict_video(dummy_video_sequence_array)

    assert isinstance(res, InferenceResult)
    assert res.is_fake is True
    assert res.label == 1
    assert res.label_name == "fake"
    assert res.fake_probability > 0.5
    assert res.num_frames == 8
    assert res.preprocessing_time_ms >= 0.0
    assert res.inference_time_ms >= 0.0
    assert res.postprocessing_time_ms >= 0.0
    assert res.total_runtime_ms >= 0.0
    assert "sequence_shape" in res.metadata


def test_inference_pipeline_single_frame_array():
    model = DummyVideoModel()
    config = VideoInferenceConfig(sequence_length=4)
    pipeline = InferencePipeline(model=model, config=config)

    single_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
    res = pipeline.predict_video(single_frame)

    assert isinstance(res, InferenceResult)
    assert res.num_frames == 4


def test_inference_pipeline_file_path_and_bytes():
    model = DummyVideoModel()
    config = VideoInferenceConfig(sequence_length=4)
    pipeline = InferencePipeline(model=model, config=config)

    # Test file path input
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(b"fake_video_bytes_content")
        tmp_path = tmp.name

    try:
        res_file = pipeline.predict_video(tmp_path)
        assert isinstance(res_file, InferenceResult)
        assert res_file.num_frames == 4
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Test bytes input
    res_bytes = pipeline.predict_video(b"mock_video_bytes")
    assert isinstance(res_bytes, InferenceResult)
    assert res_bytes.num_frames == 4


def test_inference_pipeline_predict_batch():
    model = DummyVideoModel()
    config = VideoInferenceConfig(sequence_length=4, batch_size=2)
    pipeline = InferencePipeline(model=model, config=config)

    frames1 = np.zeros((8, 64, 64, 3), dtype=np.uint8)
    frames2 = np.ones((8, 64, 64, 3), dtype=np.uint8) * 200
    frames3 = np.full((8, 64, 64, 3), 100, dtype=np.uint8)

    batch_inputs = [frames1, frames2, frames3]
    results = pipeline.predict_batch(batch_inputs, batch_size=2)

    assert len(results) == 3
    for res in results:
        assert isinstance(res, InferenceResult)
        assert res.num_frames == 4
        assert res.total_runtime_ms >= 0.0

    # Test empty batch
    empty_res = pipeline.predict_batch([])
    assert empty_res == []


def test_inference_pipeline_face_detection_fallback():
    model = DummyVideoModel()
    # Enable crop_faces True
    config = VideoInferenceConfig(sequence_length=4, crop_faces=True)
    pipeline = InferencePipeline(model=model, config=config)

    # Black frames where face detector won't find faces or falls back to center crop
    black_frames = np.zeros((4, 64, 64, 3), dtype=np.uint8)
    res = pipeline.predict_video(black_frames)

    assert isinstance(res, InferenceResult)
    assert res.num_frames == 4
    # Graceful fallback handled without crashing


def test_inference_pipeline_callable_model():
    def mock_model_callable(x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T, C, H, W]
        return torch.tensor([[0.1, 0.9]], device=x.device)

    pipeline = InferencePipeline(model=mock_model_callable)
    frames = np.zeros((4, 64, 64, 3), dtype=np.uint8)
    res = pipeline.predict_video(frames)

    assert isinstance(res, InferenceResult)
    assert res.is_fake is True
    assert res.label_name == "fake"
