"""Unit test suite for Phase 2 Audio Framework Infrastructure modules."""

from pathlib import Path
import pytest
import torch
import torch.nn as nn

from app.audio.builders.model_builder import ModelBuilder
from app.audio.builders.trainer_builder import TrainerBuilder
from app.audio.configs.model_config import AudioModelConfig
from app.audio.configs.training_config import AudioTrainingConfig
from app.audio.constants.audio_constants import (
    DEFAULT_HOP_LENGTH,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_SAMPLE_RATE,
    SUPPORTED_AUDIO_EXTENSIONS,
)
from app.audio.core.base_dataset import BaseAudioDataset
from app.audio.core.base_evaluator import BaseAudioEvaluator
from app.audio.core.base_model import BaseAudioModel
from app.audio.core.base_predictor import BaseAudioPredictor
from app.audio.core.base_preprocessor import BaseAudioPreprocessor
from app.audio.core.base_trainer import BaseAudioTrainer
from app.audio.exceptions.audio_exceptions import (
    AudioError,
    AudioFormatError,
    AudioLoadError,
    AudioProcessingError,
)
from app.audio.exceptions.dataset_exceptions import (
    CorruptAudioFileError,
    DatasetError,
    DatasetNotFoundError,
    ProtocolParsingError,
)
from app.audio.exceptions.model_exceptions import (
    ComponentNotFoundError,
    DuplicateRegistrationError,
    ModelCheckpointError,
    ModelError,
    ModelNotFoundError,
    RegistryError,
)
from app.audio.registry.base_registry import BaseRegistry
from app.audio.registry.dataset_registry import DatasetRegistry
from app.audio.registry.loss_registry import LossRegistry
from app.audio.registry.model_registry import ModelRegistry
from app.audio.registry.optimizer_registry import OptimizerRegistry
from app.audio.registry.scheduler_registry import SchedulerRegistry


class DummyAudioModel(BaseAudioModel):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TestConstants:
    def test_audio_constants(self):
        assert DEFAULT_SAMPLE_RATE == 16000
        assert DEFAULT_N_MELS == 128
        assert DEFAULT_N_FFT == 2048
        assert DEFAULT_HOP_LENGTH == 512
        assert ".wav" in SUPPORTED_AUDIO_EXTENSIONS
        assert ".flac" in SUPPORTED_AUDIO_EXTENSIONS


class TestExceptions:
    def test_audio_exceptions_hierarchy(self):
        err = AudioFormatError("invalid format")
        assert isinstance(err, AudioError)

    def test_dataset_exceptions_hierarchy(self):
        err = DatasetNotFoundError("dataset missing")
        assert isinstance(err, DatasetError)
        assert isinstance(err, AudioError)

    def test_model_exceptions_hierarchy(self):
        err = DuplicateRegistrationError("duplicate component")
        assert isinstance(err, RegistryError)
        assert isinstance(err, AudioError)


class TestCoreABCs:
    def test_base_model_instantiation_fails(self):
        with pytest.raises(TypeError):
            BaseAudioModel()

    def test_dummy_model_implementation(self, tmp_path):
        model = DummyAudioModel(num_classes=2)
        assert model.get_num_parameters() > 0
        x = torch.randn(2, 10)
        out = model(x)
        assert out.shape == (2, 2)

        save_path = tmp_path / "dummy.pth"
        model.save(save_path)
        assert save_path.exists()

        new_model = DummyAudioModel(num_classes=2)
        new_model.load(save_path)


class TestRegistries:
    def test_base_registry_register_and_get(self):
        registry = BaseRegistry[nn.Module](name="TestRegistry")
        registry.register("dummy", DummyAudioModel)
        assert "dummy" in registry.list_registered()

        retrieved = registry.get("dummy")
        assert retrieved is DummyAudioModel

    def test_base_registry_duplicate_registration_raises(self):
        registry = BaseRegistry[nn.Module](name="TestRegistry")
        registry.register("dummy", DummyAudioModel)
        with pytest.raises(DuplicateRegistrationError):
            registry.register("dummy", DummyAudioModel, overwrite=False)

    def test_base_registry_overwrite_allowed(self):
        registry = BaseRegistry[nn.Module](name="TestRegistry")
        registry.register("dummy", DummyAudioModel)
        registry.register("dummy", DummyAudioModel, overwrite=True)
        assert "dummy" in registry.list_registered()

    def test_base_registry_not_found_raises(self):
        registry = BaseRegistry[nn.Module](name="TestRegistry")
        with pytest.raises(ComponentNotFoundError):
            registry.get("nonexistent")

    def test_builtin_registries(self):
        m_reg = ModelRegistry()
        m_reg.register("dummymodel", DummyAudioModel)
        assert "dummymodel" in m_reg.list_registered()

        l_reg = LossRegistry()
        assert "cross_entropy" in l_reg.list_registered()

        o_reg = OptimizerRegistry()
        assert "adamw" in o_reg.list_registered()

        s_reg = SchedulerRegistry()
        assert "cosine" in s_reg.list_registered()


class TestBuilders:
    def test_model_builder(self):
        m_reg = ModelRegistry()
        m_reg.register("dummymodel", DummyAudioModel)

        builder = ModelBuilder(registry=m_reg)
        config = AudioModelConfig(model_name="dummymodel", num_classes=2)
        model = builder.build(config)
        assert isinstance(model, DummyAudioModel)

    def test_trainer_builder(self):
        builder = TrainerBuilder()
        model = DummyAudioModel(num_classes=2)
        config = AudioTrainingConfig(optimizer_name="adamw", scheduler_name="cosine", epochs=5)

        opt = builder.build_optimizer(model, config)
        assert isinstance(opt, torch.optim.AdamW)

        sched = builder.build_scheduler(opt, config)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

        loss_fn = builder.build_loss_function("cross_entropy")
        assert isinstance(loss_fn, nn.CrossEntropyLoss)
