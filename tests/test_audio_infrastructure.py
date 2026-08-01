"""
Unit test suite for Phase 1.2 Audio Infrastructure modules.
"""

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from app.audio.configs.dataset_config import AudioFeatureConfig, DatasetConfig
from app.audio.configs.model_config import AudioModelConfig
from app.audio.configs.training_config import AudioTrainingConfig
from app.audio.datasets.dataset_factory import DatasetFactory
from app.audio.evaluation.metrics import MetricsCalculator
from app.audio.preprocessing.pipeline import AudioPreprocessingPipeline
from app.audio.training.checkpoint import CheckpointManager
from app.audio.training.optimizer import OptimizerFactory
from app.audio.training.scheduler import SchedulerFactory
from app.audio.utils.device import DeviceManager
from app.audio.utils.logger import AudioLogger
from app.audio.utils.seed import SeedManager


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


class TestAudioConfigs:
    def test_audio_feature_config_defaults(self):
        config = AudioFeatureConfig()
        assert config.sample_rate == 16000
        assert config.n_mels == 128
        config.validate()

    def test_audio_feature_config_invalid(self):
        config = AudioFeatureConfig(sample_rate=-1)
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            config.validate()

    def test_dataset_config_post_init(self, tmp_path):
        config = DatasetConfig(dataset_dir=tmp_path, split="dev")
        assert isinstance(config.dataset_dir, Path)
        assert config.split == "dev"

    def test_audio_training_config(self):
        config = AudioTrainingConfig(batch_size=16, learning_rate=0.005)
        assert config.batch_size == 16
        assert config.learning_rate == 0.005
        config.validate()

    def test_audio_model_config(self):
        config = AudioModelConfig(num_classes=2, dropout=0.2)
        assert config.num_classes == 2
        assert config.dropout == 0.2
        config.validate()


class TestAudioUtils:
    def test_device_manager(self):
        manager = DeviceManager(requested_device="cpu")
        assert manager.device == torch.device("cpu")
        assert not manager.is_cuda
        tensor = torch.randn(2, 2)
        moved = manager.to_device(tensor)
        assert moved.device == torch.device("cpu")

    def test_seed_manager(self):
        manager = SeedManager(seed=1234)
        applied = manager.set_seed(deterministic=False)
        assert applied == 1234
        val1 = torch.randn(1).item()
        manager.set_seed(deterministic=False)
        val2 = torch.randn(1).item()
        assert val1 == val2

    def test_audio_logger(self):
        logger = AudioLogger.get("test_component")
        assert "app.audio.test_component" in logger.name


class TestDatasetFactory:
    def test_factory_initialization(self, tmp_path):
        ds_config = DatasetConfig(dataset_dir=tmp_path)
        tr_config = AudioTrainingConfig(batch_size=8)
        factory = DatasetFactory(dataset_config=ds_config, training_config=tr_config)
        assert Path(factory.dataset_config.dataset_dir).resolve() == Path(tmp_path).resolve()
        assert factory.training_config.batch_size == 8


class TestPreprocessingPipeline:
    def test_process_array(self):
        pipeline = AudioPreprocessingPipeline()
        audio = np.random.randn(16000).astype(np.float32)
        tensor = pipeline.process_array(audio, sample_rate=16000)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.ndim == 3
        assert tensor.shape[0] == 1
        assert tensor.shape[1] == 128


class TestTrainingInfrastructure:
    def test_optimizer_factory(self):
        config = AudioTrainingConfig(optimizer_name="adamw", learning_rate=1e-3)
        factory = OptimizerFactory(config)
        model = DummyModel()
        optimizer = factory.create_optimizer(model)
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_scheduler_factory(self):
        config = AudioTrainingConfig(scheduler_name="cosine", epochs=10)
        factory = SchedulerFactory(config)
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = factory.create_scheduler(optimizer)
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_checkpoint_manager(self, tmp_path):
        manager = CheckpointManager(checkpoint_dir=tmp_path, max_to_keep=2)
        model = DummyModel()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        path = manager.save(model, optimizer=opt, epoch=1, filename="chk1.pth")
        assert path.exists()

        loaded_model = DummyModel()
        chk = manager.load(path, model=loaded_model)
        assert chk["epoch"] == 1


class TestMetricsCalculator:
    def test_metrics_calculator_compute_all(self):
        calc = MetricsCalculator()
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])

        metrics = calc.compute_all(y_true, y_pred, y_scores)
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert "eer" in metrics
        assert metrics["eer"] == 0.0
