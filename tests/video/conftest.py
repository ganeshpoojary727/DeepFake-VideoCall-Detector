"""Pytest fixtures for video subsystem unit test suite."""

import pytest
import numpy as np
import torch

from app.video.configs.augmentation_config import AugmentationConfig
from app.video.configs.dataset_config import DatasetConfig
from app.video.configs.inference_config import VideoInferenceConfig
from app.video.configs.model_config import ModelConfig
from app.video.configs.training_config import VideoTrainingConfig


@pytest.fixture
def dummy_frame_array():
    """Returns a single numpy frame array [224, 224, 3]."""
    return np.zeros((224, 224, 3), dtype=np.uint8)


@pytest.fixture
def dummy_video_sequence_array():
    """Returns a video sequence array [16, 224, 224, 3]."""
    return np.zeros((16, 224, 224, 3), dtype=np.uint8)


@pytest.fixture
def dummy_video_tensor():
    """Returns a 4D video tensor [16, 3, 224, 224]."""
    return torch.zeros(16, 3, 224, 224, dtype=torch.float32)


@pytest.fixture
def dummy_batch_video_tensor():
    """Returns a 5D batch video tensor [2, 16, 3, 224, 224]."""
    return torch.zeros(2, 16, 3, 224, 224, dtype=torch.float32)


@pytest.fixture
def sample_training_config():
    """Returns VideoTrainingConfig instance."""
    return VideoTrainingConfig(epochs=2, batch_size=2)


@pytest.fixture
def sample_inference_config():
    """Returns VideoInferenceConfig instance."""
    return VideoInferenceConfig(sequence_length=8)


@pytest.fixture
def sample_dataset_config():
    """Returns DatasetConfig instance."""
    return DatasetConfig(sequence_length=8)


@pytest.fixture
def sample_augmentation_config():
    """Returns AugmentationConfig instance."""
    return AugmentationConfig()


@pytest.fixture
def sample_model_config():
    """Returns ModelConfig instance."""
    return ModelConfig(num_classes=2, sequence_length=8)
