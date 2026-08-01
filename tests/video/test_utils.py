"""Unit tests for video utilities."""

import os
import tempfile
import numpy as np
import pytest
import torch

from app.video.models.video_factory import VideoFactory
from app.video.utils.checkpoint_utils import (
    inspect_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from app.video.utils.device import DeviceManager, get_device
from app.video.utils.logger import VideoLogger, get_video_logger
from app.video.utils.seed import SeedManager, set_seed
from app.video.utils.visualization import (
    draw_bboxes,
    plot_training_curves,
    visualize_frames,
)


def test_logger():
    logger = get_video_logger("TestLogger")
    assert logger is not None
    assert logger.name == "TestLogger"


def test_seed():
    set_seed(42)
    t1 = torch.randn(2)
    set_seed(42)
    t2 = torch.randn(2)
    assert torch.equal(t1, t2)

    manager = SeedManager(default_seed=123)
    assert manager.current_seed == 123


def test_device():
    dev = get_device("cpu")
    assert dev.type == "cpu"

    manager = DeviceManager(preferred="cpu")
    assert manager.device.type == "cpu"
    t = manager.to_device(torch.zeros(2))
    assert t.device.type == "cpu"


def test_checkpoint_utils():
    model = VideoFactory.create_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_ckpt.pt")
        save_checkpoint(filepath=path, model=model, epoch=5, metrics={"acc": 0.9})
        assert os.path.exists(path)

        info = inspect_checkpoint(path)
        assert info["epoch"] == 5
        assert info["metrics"]["acc"] == 0.9

        payload = load_checkpoint(filepath=path, model=model)
        assert payload["epoch"] == 5


def test_visualization(dummy_frame_array, dummy_video_tensor):
    bboxes = [(10, 10, 50, 50)]
    annotated = draw_bboxes(dummy_frame_array, bboxes)
    assert annotated.shape == dummy_frame_array.shape

    strip = visualize_frames(dummy_video_tensor, max_frames=4)
    assert strip.ndim == 3
    assert strip.shape[1] == 224 * 4  # Tile 4 frames horizontally

    curves = plot_training_curves([0.5, 0.4], [0.6, 0.5])
    assert "train_loss" in curves
    assert "val_loss" in curves
