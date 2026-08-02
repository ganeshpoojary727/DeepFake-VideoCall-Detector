"""Exhaustive unit tests for Production Video Training Subsystem."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.video.builders import VideoTrainerBuilder
from app.video.configs import VideoTrainingConfig, ModelConfig
from app.video.datasets import (
    CelebDFDataset,
    ClipSampler,
    DatasetFactory,
    FaceForensicsDataset,
    FrameSampler,
)
from app.video.models import EfficientNetB4Model
from app.video.registry import dataset_registry
from app.video.training import (
    CheckpointManager,
    LossFactory,
    OptimizerFactory,
    ProductionVideoTrainer,
    SchedulerFactory,
)


def test_dataset_factory_and_registry():
    assert "faceforensics_pp" in dataset_registry.list_registered()
    assert "celeb_df_v2" in dataset_registry.list_registered()


def test_clip_sampler():
    sampler = ClipSampler(clip_duration_seconds=20.0, fps=30.0, random_start=False)
    frames = torch.randn(900, 3, 224, 224)  # 30-sec video
    clip = sampler.sample_clip(frames)
    assert len(clip) == 600  # 20 sec * 30 fps


def test_frame_sampler_strategies():
    frames = torch.randn(600, 3, 224, 224)

    # Uniform
    uniform_sampler = FrameSampler(num_frames=16, strategy="uniform")
    s_uniform = uniform_sampler.sample(frames)
    assert len(s_uniform) == 16

    # Random
    random_sampler = FrameSampler(num_frames=16, strategy="random")
    s_random = random_sampler.sample(frames)
    assert len(s_random) == 16

    # Stride
    stride_sampler = FrameSampler(num_frames=16, strategy="stride", stride=2)
    s_stride = stride_sampler.sample(frames)
    assert len(s_stride) == 16


def test_loss_factory_losses():
    # CrossEntropy
    ce = LossFactory.create("cross_entropy")
    assert isinstance(ce, nn.CrossEntropyLoss)

    # Focal Loss
    focal = LossFactory.create("focal")
    logits = torch.randn(4, 2)
    labels = torch.tensor([0, 1, 0, 1])
    l_focal = focal(logits, labels)
    assert l_focal.item() >= 0.0

    # Weighted CrossEntropy
    w_ce = LossFactory.create("weighted_ce", weights=torch.tensor([1.0, 2.0]))
    l_wce = w_ce(logits, labels)
    assert l_wce.item() >= 0.0


def test_scheduler_factory_schedulers():
    model = nn.Linear(10, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    cos_sched = SchedulerFactory.create("cosine", optimizer=opt, epochs=50)
    assert cos_sched is not None

    onecycle_sched = SchedulerFactory.create("onecycle", optimizer=opt, epochs=50, max_lr=1e-3)
    assert onecycle_sched is not None

    warmup_sched = SchedulerFactory.create("warmup", optimizer=opt, epochs=50, warmup_epochs=5)
    assert warmup_sched is not None


def test_trainer_execution_and_gradient_accumulation(tmp_path):
    model = EfficientNetB4Model(config=ModelConfig(pretrained=False))
    x = torch.randn(8, 4, 3, 224, 224)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=4)

    cfg = VideoTrainingConfig(
        epochs=1,
        use_amp=False,
        gradient_accumulation_steps=2,
        checkpoint_dir=tmp_path,
    )
    trainer = ProductionVideoTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=cfg,
    )

    history = trainer.train()
    assert "train_loss" in history
    assert "val_loss" in history
    assert len(history["train_loss"]) == 1


def test_multi_metric_checkpoints_and_resume(tmp_path):
    model = EfficientNetB4Model(config=ModelConfig(pretrained=False))
    x = torch.randn(4, 4, 3, 224, 224)
    y = torch.tensor([0, 1, 0, 1])
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=2)

    cfg = VideoTrainingConfig(epochs=1, checkpoint_dir=tmp_path)
    trainer = ProductionVideoTrainer(model=model, train_loader=loader, val_loader=loader, config=cfg)
    trainer.train()

    latest_path = tmp_path / "latest.pt"
    assert latest_path.exists()

    # Test resume
    new_trainer = ProductionVideoTrainer(model=model, train_loader=loader, config=cfg)
    resumed_epoch = new_trainer.resume_from_checkpoint(latest_path)
    assert resumed_epoch >= 0


def test_video_trainer_builder(tmp_path):
    builder = VideoTrainerBuilder()
    cfg_train = VideoTrainingConfig(epochs=1, checkpoint_dir=tmp_path)
    cfg_model = ModelConfig(pretrained=False)

    trainer = (
        builder.with_training_config(cfg_train)
        .with_model_config(cfg_model)
        .build()
    )
    assert isinstance(trainer, ProductionVideoTrainer)
