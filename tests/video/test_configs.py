"""Unit tests for video configuration classes."""

import pytest
from app.video.configs.augmentation_config import AugmentationConfig
from app.video.configs.dataset_config import DatasetConfig
from app.video.configs.inference_config import VideoInferenceConfig
from app.video.configs.model_config import ModelConfig
from app.video.configs.training_config import VideoTrainingConfig
from app.video.exceptions.video_exceptions import ConfigurationError


def test_video_training_config_defaults():
    config = VideoTrainingConfig()
    assert config.epochs == 50
    assert config.batch_size == 8
    assert config.learning_rate == 1e-4
    config.validate()


def test_video_training_config_invalid_epochs():
    config = VideoTrainingConfig(epochs=-1)
    with pytest.raises(ConfigurationError):
        config.validate()


def test_video_training_config_invalid_batch_size():
    config = VideoTrainingConfig(batch_size=0)
    with pytest.raises(ConfigurationError):
        config.validate()


def test_video_training_config_invalid_lr():
    config = VideoTrainingConfig(learning_rate=-0.01)
    with pytest.raises(ConfigurationError):
        config.validate()


def test_video_training_config_extra_params():
    config = VideoTrainingConfig(extra_params={"custom_key": 123})
    assert config.extra_params["custom_key"] == 123
    config.validate()


def test_video_inference_config_defaults():
    config = VideoInferenceConfig()
    assert config.target_fps == 30.0
    assert config.sequence_length == 16
    config.validate()


def test_video_inference_config_invalid_seq_len():
    config = VideoInferenceConfig(sequence_length=0)
    with pytest.raises(ConfigurationError):
        config.validate()


def test_video_inference_config_invalid_threshold():
    config = VideoInferenceConfig(confidence_threshold=1.5)
    with pytest.raises(ConfigurationError):
        config.validate()


def test_video_inference_config_extra_params():
    config = VideoInferenceConfig(extra_params={"debug": True})
    assert config.extra_params["debug"] is True
    config.validate()


def test_dataset_config_defaults():
    config = DatasetConfig()
    assert config.dataset_name == "faceforensics_pp"
    assert config.sequence_length == 16
    config.validate()


def test_dataset_config_empty_name():
    config = DatasetConfig(dataset_name="")
    with pytest.raises(ConfigurationError):
        config.validate()


def test_dataset_config_invalid_seq_len():
    config = DatasetConfig(sequence_length=0)
    with pytest.raises(ConfigurationError):
        config.validate()


def test_augmentation_config_defaults():
    config = AugmentationConfig()
    assert config.enable_augmentation is True
    config.validate()


def test_augmentation_config_invalid_jpeg():
    config = AugmentationConfig(jpeg_quality_range=(100, 50))
    with pytest.raises(ConfigurationError):
        config.validate()


def test_augmentation_config_invalid_flip():
    config = AugmentationConfig(horizontal_flip_prob=2.0)
    with pytest.raises(ConfigurationError):
        config.validate()


def test_model_config_defaults():
    config = ModelConfig()
    assert config.model_name == "video_detector"
    assert config.num_classes == 2
    config.validate()


def test_model_config_invalid_classes():
    config = ModelConfig(num_classes=0)
    with pytest.raises(ConfigurationError):
        config.validate()


def test_model_config_invalid_channels():
    config = ModelConfig(in_channels=0)
    with pytest.raises(ConfigurationError):
        config.validate()


def test_model_config_invalid_dropout():
    config = ModelConfig(dropout=1.5)
    with pytest.raises(ConfigurationError):
        config.validate()
