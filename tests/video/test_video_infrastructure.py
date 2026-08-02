"""Exhaustive unit tests for video AI framework infrastructure."""

import os
from pathlib import Path
import pytest
import numpy as np
import torch
import torch.nn as nn

from app.video.constants import (
    DATASET_FACEFORENSICS,
    DATASET_CELEBDFV2,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    IMAGENET_MEAN,
    IMAGENET_STD,
    VIDEO_CACHE_DIR,
)
from app.video.exceptions import (
    VideoError,
    DatasetNotFound,
    FrameExtractionError,
    FaceDetectionError,
    ModelInitializationError,
    CheckpointError,
    RegistryError,
    ComponentNotFoundError,
    DuplicateRegistrationError,
)
from app.video.configs import (
    DatasetConfig,
    ModelConfig,
    VideoTrainingConfig,
    AugmentationConfig,
    VideoInferenceConfig,
)
from app.video.registry import (
    ModelRegistry,
    DatasetRegistry,
    OptimizerRegistry,
    SchedulerRegistry,
    AugmentationRegistry,
    LossRegistry,
)
from app.video.builders import (
    VideoModelBuilder,
    DatasetBuilder,
    TrainingBuilder,
    AugmentationBuilder,
    VideoPipelineBuilder,
)
from app.video.core import (
    BaseVideoModel,
    BaseDataset,
    BaseTrainer,
    BaseEvaluator,
    BaseInferenceEngine,
    BaseFeatureExtractor,
)
from app.video.datasets import (
    DatasetScanner,
    DatasetIndexBuilder,
    ProductionVideoDataLoader,
    video_collate_fn,
)
from app.video.augmentation import (
    HorizontalFlip,
    RandomCrop,
    RandomRotation,
    ColorJitter,
    GaussianBlur,
    JPEGCompression,
    GaussianNoise,
    FrameDropout,
    TemporalJitter,
    VideoAugmentationPipeline,
)
from app.video.preprocessing import (
    FrameExtractor,
    VideoDecoder,
    VideoNormalizer,
    FaceCropper,
    FaceAligner,
    FrameSampler,
    FrameCache,
    VideoPreprocessor,
)
from app.video.features import SpatialFeatureExtractor, TemporalFeatureExtractor
from app.video.training import (
    ProductionVideoTrainer,
    CheckpointManager,
    EarlyStopping,
    MixedPrecisionHandler,
    LossFactory,
    OptimizerFactory,
    SchedulerFactory,
)
from app.video.evaluation import (
    EvaluationMetrics,
    ConfusionMatrix,
    PerformanceEvaluator,
    VideoEvaluator,
)
from app.video.inference import (
    WindowCapture,
    FrameQueue,
    VideoDetector,
    InferencePostProcessor,
)
from app.video.pipeline import InferencePipeline


# Dummy Model for Testing
class DummyModel(BaseVideoModel):
    def __init__(self, num_classes: int = 2, in_channels: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)
            out = self.fc(self.pool(self.conv(x)).view(b * t, -1))
            out = out.view(b, t, -1).mean(dim=1)
            return out
        else:
            return self.fc(self.pool(self.conv(x)).view(x.size(0), -1))

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test Configurations
def test_video_configurations():
    ds_cfg = DatasetConfig(dataset_name=DATASET_FACEFORENSICS, sequence_length=16)
    assert ds_cfg.dataset_name == DATASET_FACEFORENSICS
    assert ds_cfg.sequence_length == 16
    ds_cfg.validate()

    m_cfg = ModelConfig(num_classes=2)
    assert m_cfg.num_classes == 2
    m_cfg.validate()

    tr_cfg = VideoTrainingConfig(epochs=5, batch_size=8)
    assert tr_cfg.epochs == 5
    tr_cfg.validate()

    aug_cfg = AugmentationConfig(enable_augmentation=True)
    aug_cfg.validate()

    inf_cfg = VideoInferenceConfig(sequence_length=16)
    inf_cfg.validate()


# Test Exceptions
def test_video_exceptions():
    err = VideoError("Base error")
    assert isinstance(err, Exception)

    ds_err = DatasetNotFound("Dataset missing")
    assert isinstance(ds_err, VideoError)

    fe_err = FrameExtractionError("Extraction failed")
    assert isinstance(fe_err, VideoError)

    chk_err = CheckpointError("Checkpoint failed")
    assert isinstance(chk_err, VideoError)


# Test Registries
def test_video_registries():
    reg = ModelRegistry()
    reg.register("dummy", DummyModel)
    assert "dummy" in reg.list_registered()
    retrieved = reg.get("dummy")
    assert retrieved == DummyModel

    with pytest.raises(DuplicateRegistrationError):
        reg.register("dummy", DummyModel, overwrite=False)

    with pytest.raises(ComponentNotFoundError):
        reg.get("nonexistent")


# Test Builders
def test_video_builders():
    m_builder = VideoModelBuilder()
    m_builder.registry.register("dummy_builder", DummyModel, overwrite=True)
    m_cfg = ModelConfig(model_name="dummy_builder")
    model = m_builder.build(m_cfg)
    assert isinstance(model, nn.Module)

    aug_builder = AugmentationBuilder()
    aug_pipe = aug_builder.build(AugmentationConfig(enable_augmentation=True))
    assert isinstance(aug_pipe, VideoAugmentationPipeline)

    p_builder = VideoPipelineBuilder()
    t_pipe = p_builder.with_model(model).build_training_pipeline()
    assert t_pipe is not None


# Test Dataset Discovery & Indexing
def test_dataset_discovery_and_indexing(tmp_path):
    scanner = DatasetScanner(root_dir=tmp_path)
    reports = scanner.scan_all()
    assert DATASET_FACEFORENSICS in reports
    assert DATASET_CELEBDFV2 in reports

    builder = DatasetIndexBuilder(data_dir=tmp_path, cache_dir=tmp_path / "cache")
    items = builder.build_or_load_index(DATASET_FACEFORENSICS, split="train")
    assert len(items) > 0
    assert os.path.exists(builder.get_cache_path(DATASET_FACEFORENSICS, "train"))
    stats = builder.get_statistics(items)
    assert stats["total_samples"] == len(items)


# Test DataLoader
def test_production_dataloader():
    mock_batch = [
        {"tensor": torch.randn(16, 3, 224, 224), "label": 0, "filepath": "path1"},
        {"tensor": torch.randn(16, 3, 224, 224), "label": 1, "filepath": "path2"},
    ]
    collated = video_collate_fn(mock_batch)
    assert collated["tensor"].shape == (2, 16, 3, 224, 224)
    assert collated["label"].shape == (2,)


# Test Augmentation Framework
def test_augmentation_framework():
    dummy_video = torch.rand(16, 3, 224, 224)
    hflip = HorizontalFlip(p=1.0)
    out_hflip = hflip(dummy_video)
    assert out_hflip.shape == dummy_video.shape

    crop = RandomCrop(crop_size=(112, 112), p=1.0)
    out_crop = crop(dummy_video)
    assert out_crop.shape[-2:] == (112, 112)

    noise = GaussianNoise(std=0.01, p=1.0)
    out_noise = noise(dummy_video)
    assert out_noise.shape == dummy_video.shape

    drop = FrameDropout(drop_prob=0.2, p=1.0)
    out_drop = drop(dummy_video)
    assert out_drop.shape == dummy_video.shape


# Test Preprocessing Interfaces
def test_preprocessing_interfaces():
    extractor = FrameExtractor(max_frames=8)
    arr = np.zeros((16, 224, 224, 3), dtype=np.uint8)
    frames = extractor.extract(arr)
    assert len(frames) == 8

    decoder = VideoDecoder()
    decoded = decoder.decode(arr)
    assert len(decoded) == 16

    normalizer = VideoNormalizer()
    tensor_video = torch.rand(16, 3, 224, 224)
    normed = normalizer.normalize(tensor_video)
    assert normed.shape == tensor_video.shape

    cache = FrameCache(max_memory_items=5)
    cache.put("key1", frames)
    assert cache.get("key1") is not None


# Test Trainer & Loss/Optimizer/Scheduler Factories
def test_trainer_and_factories(tmp_path):
    model = DummyModel()
    t_cfg = VideoTrainingConfig(epochs=1, batch_size=2, checkpoint_dir=tmp_path)
    opt = OptimizerFactory.create_optimizer(model, config=t_cfg)
    loss_fn = LossFactory.create_loss(config=t_cfg)
    sched = SchedulerFactory.create_scheduler(opt, config=t_cfg)

    assert opt is not None
    assert loss_fn is not None
    assert sched is not None

    trainer = ProductionVideoTrainer(
        model=model,
        config=t_cfg,
        optimizer=opt,
        loss_fn=loss_fn,
        scheduler=sched,
    )
    assert trainer.device is not None


# Test Evaluation Framework
def test_evaluation_metrics():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_probs = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
    metrics = EvaluationMetrics.compute_all(y_true, y_probs)
    assert metrics["accuracy"] == 1.0
    assert metrics["auc"] > 0.8

    cm = ConfusionMatrix(y_true, (y_probs >= 0.5).astype(int))
    assert cm.tp == 3
    assert cm.tn == 3

    gpu_stats = PerformanceEvaluator.measure_gpu_memory()
    assert "allocated_mb" in gpu_stats


# Test Real-Time Inference Interfaces
def test_inference_interfaces():
    capture = WindowCapture(window_size=4, stride=1)
    win = None
    for _ in range(5):
        win = capture.add_frame(np.zeros((224, 224, 3), dtype=np.uint8))
    assert win is not None and len(win) == 4

    queue = FrameQueue(maxsize=10)
    assert queue.put(np.zeros((224, 224, 3), dtype=np.uint8))
    assert queue.get() is not None

    detector = VideoDetector()
    assert detector.is_ready

    post = InferencePostProcessor()
    res = post.process_outputs(torch.tensor([[0.2, 0.8]]))
    assert res["label_name"] == "fake"
