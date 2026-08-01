"""Unit tests for high-level pipelines and fluent builder pattern."""

import numpy as np
import pytest
from app.video.builders.video_builder import VideoPipelineBuilder
from app.video.configs.inference_config import VideoInferenceConfig
from app.video.configs.model_config import ModelConfig
from app.video.configs.training_config import VideoTrainingConfig
from app.video.datasets.video_dataset import VideoDataset
from app.video.models.video_factory import VideoFactory
from app.video.pipeline.inference_pipeline import InferencePipeline
from app.video.pipeline.training_pipeline import TrainingPipeline
from app.video.pipeline.validation_pipeline import ValidationPipeline


def test_training_pipeline_run():
    m_cfg = ModelConfig(sequence_length=4)
    t_cfg = VideoTrainingConfig(epochs=1, batch_size=2)
    pipeline = TrainingPipeline(training_config=t_cfg, model_config=m_cfg)

    train_ds = VideoDataset(samples=[{"label": 0}, {"label": 1}])
    val_ds = VideoDataset(samples=[{"label": 0}, {"label": 1}])

    history = pipeline.run(train_ds, val_ds)
    assert "train_loss" in history
    assert "val_loss" in history
    assert len(history["train_loss"]) == 1


def test_validation_pipeline_evaluate():
    model = VideoFactory.create_model()
    pipeline = ValidationPipeline(model=model)
    val_ds = VideoDataset(samples=[{"label": 0}, {"label": 1}])
    metrics = pipeline.evaluate(val_ds, batch_size=2)
    assert "accuracy" in metrics


def test_inference_pipeline_predict_array(dummy_video_sequence_array):
    model = VideoFactory.create_model()
    i_cfg = VideoInferenceConfig(sequence_length=16)
    pipeline = InferencePipeline(model=model, config=i_cfg)

    res = pipeline.predict_video(dummy_video_sequence_array)
    assert "is_fake" in res
    assert "fake_probability" in res
    assert "real_probability" in res
    assert isinstance(res["is_fake"], bool)


def test_video_pipeline_builder():
    builder = (
        VideoPipelineBuilder()
        .with_model_config(ModelConfig(sequence_length=4))
        .with_training_config(VideoTrainingConfig(epochs=1))
        .with_inference_config(VideoInferenceConfig(sequence_length=4))
    )

    t_pipe = builder.build_training_pipeline()
    assert isinstance(t_pipe, TrainingPipeline)

    i_pipe = builder.build_inference_pipeline()
    assert isinstance(i_pipe, InferencePipeline)
