"""Exhaustive unit tests for production AASIST audio training pipeline."""

import os
import tempfile
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from app.audio.augmentation.audio_augmentations import (
    BackgroundNoise,
    CompressionSimulation,
    FrequencyMasking,
    Gain,
    GaussianNoise,
    RandomCropping,
    RandomShift,
    Reverberation,
    SpecAugment,
    TimeMasking,
)
from app.audio.augmentation.augmentation_pipeline import AudioAugmentationPipeline
from app.audio.configs.data_config import AudioDataConfig
from app.audio.configs.pipeline_config import AudioPipelineConfig
from app.audio.configs.training_config import AudioTrainingConfig
from app.audio.datasets.dataset_scanner import DatasetScanner
from app.audio.datasets.index_builder import DatasetIndexBuilder
from app.audio.datasets.dataloader import ProductionAudioDataLoader, audio_collate_fn
from app.audio.models.aasist.model import AASIST
from app.audio.pipeline.audio_pipeline import AudioPipeline
from app.audio.training.checkpoint import CheckpointManager
from app.audio.training.eer_metrics import compute_biometric_metrics, compute_eer
from app.audio.training.ema import EMAModel
from app.audio.training.loss_factory import ClassBalancedLoss, FocalLoss, LossFactory
from app.audio.training.metrics import AudioMetricsCalculator
from app.audio.training.optimizer import OptimizerFactory
from app.audio.training.optimizers.lion_optimizer import Lion
from app.audio.training.scheduler import SchedulerFactory, WarmupCosineLR
from app.audio.training.trainer import ProductionAudioTrainer
from app.audio.training.validator import ValidationEngine


# Dummy dataset for dataloader & trainer tests
class MockAudioDataset(Dataset):
    def __init__(self, count: int = 10, length: int = 64600) -> None:
        self.count = count
        self.length = length

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int) -> dict:
        return {
            "tensor": torch.randn(1, self.length),
            "label": idx % 2,
            "sample_path": f"mock_{idx}.flac",
        }


def test_dataset_scanner():
    scanner = DatasetScanner()
    report = scanner.scan_dataset("asvspoof2019_la")
    assert report.dataset_name == "asvspoof2019_la"
    assert isinstance(report.missing_items, list)


def test_dataset_index_builder():
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = DatasetIndexBuilder(cache_dir=tmpdir)
        items = builder.build_or_load_index("asvspoof2019_la", "train")
        assert len(items) > 0
        assert items[0].dataset == "asvspoof2019_la"
        cache_path = builder.get_cache_path("asvspoof2019_la", "train")
        assert cache_path.exists()


def test_production_dataloader():
    ds = MockAudioDataset(count=4)
    cfg = AudioDataConfig(batch_size=2, num_workers=0)
    loader = ProductionAudioDataLoader.create_dataloader(ds, config=cfg)
    batch = next(iter(loader))
    assert "tensor" in batch
    assert "label" in batch
    assert batch["tensor"].shape == (2, 1, 64600)


def test_audio_pipeline():
    cfg = AudioPipelineConfig(target_samples=64600, normalize=True)
    pipe = AudioPipeline(config=cfg)
    raw = np.random.randn(50000).astype(np.float32)
    out = pipe(raw)
    assert out.shape == (1, 64600)


def test_audio_augmentations_independently():
    x = torch.randn(1, 64600)
    aug1 = GaussianNoise(p=1.0)
    aug2 = BackgroundNoise(p=1.0)
    aug3 = Gain(p=1.0)
    aug4 = TimeMasking(p=1.0)
    aug5 = FrequencyMasking(p=1.0)
    aug6 = SpecAugment(p=1.0)
    aug7 = RandomCropping(target_samples=30000, p=1.0)
    aug8 = RandomShift(p=1.0)
    aug9 = Reverberation(p=1.0)
    aug10 = CompressionSimulation(p=1.0)

    aug1.train()
    aug2.train()
    aug3.train()
    aug4.train()
    aug5.train()
    aug6.train()
    aug7.train()
    aug8.train()
    aug9.train()
    aug10.train()

    assert aug1(x).shape == x.shape
    assert aug2(x).shape == x.shape
    assert aug3(x).shape == x.shape
    assert aug4(x).shape == x.shape
    assert aug5(x).shape == x.shape
    assert aug6(x).shape == x.shape
    assert aug7(x).shape == (1, 30000)
    assert aug8(x).shape == x.shape
    assert aug9(x).shape == x.shape
    assert aug10(x).shape == x.shape


def test_augmentation_pipeline():
    pipeline = AudioAugmentationPipeline(
        transforms=[GaussianNoise(p=1.0), Gain(p=1.0)]
    )
    pipeline.train()
    x = torch.randn(1, 64600)
    out = pipeline(x)
    assert out.shape == x.shape


def test_loss_factory():
    cfg = AudioTrainingConfig(loss_name="focal")
    factory = LossFactory(cfg)
    loss_fn = factory.create_loss()
    assert isinstance(loss_fn, FocalLoss)

    cfg2 = AudioTrainingConfig(loss_name="class_balanced")
    loss_fn2 = LossFactory(cfg2).create_loss()
    assert isinstance(loss_fn2, ClassBalancedLoss)


def test_optimizer_factory():
    model = AASIST()
    cfg = AudioTrainingConfig(optimizer_name="lion")
    opt = OptimizerFactory(cfg).create_optimizer(model)
    assert isinstance(opt, Lion)


def test_scheduler_factory():
    model = AASIST()
    cfg = AudioTrainingConfig(scheduler_name="warmup_cosine")
    opt = OptimizerFactory(cfg).create_optimizer(model)
    sched = SchedulerFactory(cfg).create_scheduler(opt)
    assert isinstance(sched, WarmupCosineLR)


def test_biometric_metrics():
    probs = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    labels = torch.tensor([0, 1])
    bio = compute_biometric_metrics(probs, labels)
    assert "eer" in bio
    assert "hter" in bio
    assert "apcer" in bio
    assert "bpcer" in bio


def test_audio_metrics_calculator():
    logits = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])
    labels = torch.tensor([0, 1])
    metrics = AudioMetricsCalculator.compute_all(logits, labels, latency_ms=12.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["latency_ms"] == 12.5


def test_ema_model():
    model = AASIST()
    ema = EMAModel(model, decay=0.99)
    ema.update(model)
    assert isinstance(ema.state_dict(), dict)


def test_validation_engine():
    model = AASIST()
    ds = MockAudioDataset(count=2)
    loader = DataLoader(ds, batch_size=2, collate_fn=audio_collate_fn)
    engine = ValidationEngine(model=model, device="cpu")
    metrics = engine.evaluate(loader)
    assert "accuracy" in metrics
    assert "eer" in metrics


def test_trainer_execution():
    model = AASIST()
    ds_train = MockAudioDataset(count=4)
    ds_val = MockAudioDataset(count=2)
    loader_train = DataLoader(ds_train, batch_size=2, collate_fn=audio_collate_fn)
    loader_val = DataLoader(ds_val, batch_size=2, collate_fn=audio_collate_fn)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = AudioTrainingConfig(
            epochs=1,
            batch_size=2,
            checkpoint_dir=tmpdir,
            log_dir=tmpdir,
            tensorboard_dir=os.path.join(tmpdir, "tb"),
            use_amp=False,
        )
        trainer = ProductionAudioTrainer(
            model=model,
            train_loader=loader_train,
            val_loader=loader_val,
            config=cfg,
        )
        history = trainer.train()
        assert "train_loss" in history
        assert os.path.exists(os.path.join(tmpdir, "best_model.pt"))
        assert os.path.exists(os.path.join(tmpdir, "last_checkpoint.pt"))


def test_aasist_forward_pass():
    model = AASIST()
    x = torch.randn(2, 1, 64600)
    logits = model(x)
    assert logits.shape == (2, 2)
    num_params = model.get_num_parameters()
    assert num_params > 0
