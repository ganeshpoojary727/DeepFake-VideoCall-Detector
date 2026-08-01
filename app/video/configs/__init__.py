"""Video configs package."""

from app.video.configs.augmentation_config import AugmentationConfig
from app.video.configs.dataset_config import DatasetConfig
from app.video.configs.inference_config import VideoInferenceConfig
from app.video.configs.model_config import ModelConfig
from app.video.configs.training_config import VideoTrainingConfig

__all__ = [
    "VideoTrainingConfig",
    "VideoInferenceConfig",
    "DatasetConfig",
    "AugmentationConfig",
    "ModelConfig",
]
