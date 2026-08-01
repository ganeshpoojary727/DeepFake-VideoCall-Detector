"""Unit tests for training metrics, callbacks, factories, trainer, and checkpoint manager."""

import os
import tempfile
import pytest
import torch
from torch.utils.data import DataLoader

from app.video.configs.training_config import VideoTrainingConfig
from app.video.datasets.video_dataset import VideoDataset
from app.video.datasets.video_sample import video_collate_fn
from app.video.models.video_factory import VideoFactory
from app.video.training.callbacks import BaseCallback, CallbackHandler, LoggingCallback
from app.video.training.checkpoint_manager import CheckpointManager
from app.video.training.early_stopping import EarlyStopping
from app.video.training.loss_factory import FocalLoss, LossFactory
from app.video.training.metrics import VideoMetricsCalculator, calculate_video_metrics
from app.video.training.optimizer_factory import OptimizerFactory
from app.video.training.scheduler_factory import SchedulerFactory
from app.video.training.trainer import Trainer


def test_metrics_calculator():
    y_pred = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
    y_true = torch.tensor([0, 1])
    metrics = VideoMetricsCalculator.compute_all(y_pred, y_true)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_early_stopping():
    es = EarlyStopping(patience=2, min_delta=0.01)
    assert not es(1.0)
    assert not es(0.999)  # Small improvement
    assert es(0.999)  # Patience 2 reached
    assert es.should_stop is True


def test_focal_loss():
    fl = FocalLoss()
    logits = torch.randn(2, 2)
    targets = torch.tensor([0, 1])
    loss = fl(logits, targets)
    assert loss.dim() == 0
    assert loss.item() >= 0.0


def test_optimizer_factory():
    model = VideoFactory.create_model()
    opt = OptimizerFactory.create("adamw", model.parameters(), lr=1e-3)
    assert opt is not None


def test_scheduler_factory():
    model = VideoFactory.create_model()
    opt = OptimizerFactory.create("adam", model.parameters())
    sched = SchedulerFactory.create("cosine", opt, epochs=10)
    assert sched is not None


def test_loss_factory():
    loss = LossFactory.create("cross_entropy")
    assert loss is not None


def test_checkpoint_manager():
    model = VideoFactory.create_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir, save_top_k=2)
        p1 = manager.save(model, epoch=1, metrics={"val_loss": 0.5})
        p2 = manager.save(model, epoch=2, metrics={"val_loss": 0.3})
        p3 = manager.save(model, epoch=3, metrics={"val_loss": 0.1})
        assert os.path.exists(p3)


def test_trainer_execution():
    model = VideoFactory.create_model()
    cfg = VideoTrainingConfig(epochs=1, batch_size=2)
    trainer = Trainer(model=model, config=cfg)

    ds = VideoDataset(samples=[{"label": 0}, {"label": 1}])
    loader = DataLoader(ds, batch_size=2, collate_fn=video_collate_fn)

    loss = trainer.train_epoch(loader)
    assert loss >= 0.0

    val_metrics = trainer.validate(loader)
    assert "val_loss" in val_metrics
    assert "accuracy" in val_metrics
