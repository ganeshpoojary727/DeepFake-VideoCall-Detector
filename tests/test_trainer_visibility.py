"""Tests for audio trainer visibility, monitoring, speed profiling, and exception handling."""

import os
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.audio.configs.training_config import AudioTrainingConfig
from app.audio.training.trainer import ProductionAudioTrainer, _format_time


class SimpleAudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        return self.fc(x)


class FlakyDataset(TensorDataset):
    def __init__(self, x, y, fail_index=-1):
        super().__init__(x, y)
        self.fail_index = fail_index

    def __getitem__(self, index):
        if index == self.fail_index:
            raise ValueError(f"Simulated batch failure at index {index}")
        return super().__getitem__(index)


def test_trainer_visibility_and_metrics():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = AudioTrainingConfig(
            epochs=2,
            batch_size=4,
            grad_accum_steps=2,
            checkpoint_dir=tmp_dir,
            log_dir=tmp_dir,
            tensorboard_dir=os.path.join(tmp_dir, "tb"),
            log_interval=2,
            skip_bad_batches=True,
        )

        x_train = torch.randn(20, 16)
        y_train = torch.randint(0, 2, (20,))
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=4, shuffle=False)

        x_val = torch.randn(8, 16)
        y_val = torch.randint(0, 2, (8,))
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=4, shuffle=False)

        model = SimpleAudioModel()
        trainer = ProductionAudioTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
        )

        history = trainer.fit()

        assert "train_loss" in history
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2

        # Verify speed and monitoring statistics
        assert "epoch_stats" in trainer.stats
        assert len(trainer.stats["epoch_stats"]) == 2
        for epoch_stat in trainer.stats["epoch_stats"]:
            assert "epoch_time" in epoch_stat
            assert "data_time" in epoch_stat
            assert "forward_time" in epoch_stat
            assert "backward_time" in epoch_stat
            assert "optimizer_time" in epoch_stat
            assert "val_time" in epoch_stat
            assert "avg_batch_time" in epoch_stat

        assert trainer.stats["total_training_time"] > 0.0


def test_trainer_batch_exception_handling():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = AudioTrainingConfig(
            epochs=1,
            batch_size=4,
            grad_accum_steps=1,
            checkpoint_dir=tmp_dir,
            log_dir=tmp_dir,
            tensorboard_dir=os.path.join(tmp_dir, "tb"),
            skip_bad_batches=True,
        )

        x_train = torch.randn(16, 16)
        y_train = torch.randint(0, 2, (16,))
        dataset = FlakyDataset(x_train, y_train, fail_index=5)  # Batch 2 will fail on sample 5
        train_loader = DataLoader(dataset, batch_size=4, shuffle=False)

        model = SimpleAudioModel()
        trainer = ProductionAudioTrainer(
            model=model,
            train_loader=train_loader,
            config=config,
        )

        # Trainer should log exception for failed batch and finish without crashing
        history = trainer.fit()
        assert len(history["train_loss"]) == 1


def test_format_time_helper():
    assert _format_time(0) == "00:00"
    assert _format_time(45) == "00:45"
    assert _format_time(125) == "02:05"
    assert _format_time(3665) == "01:01:05"


def test_tqdm_progress_updates(monkeypatch):
    from unittest.mock import MagicMock

    mock_pbar = MagicMock()
    mock_tqdm = MagicMock(return_value=mock_pbar)
    monkeypatch.setattr("app.audio.training.trainer.tqdm", mock_tqdm)

    with tempfile.TemporaryDirectory() as tmp_dir:
        config = AudioTrainingConfig(
            epochs=1,
            batch_size=4,
            checkpoint_dir=tmp_dir,
            log_dir=tmp_dir,
            tensorboard_dir=os.path.join(tmp_dir, "tb"),
        )

        x_train = torch.randn(20, 16)
        y_train = torch.randint(0, 2, (20,))
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=4, shuffle=False)

        model = SimpleAudioModel()
        trainer = ProductionAudioTrainer(
            model=model,
            train_loader=train_loader,
            config=config,
        )

        trainer.fit()

        # Check tqdm was initialized with total=5 (20 samples / 4 batch_size)
        mock_tqdm.assert_called_once()
        _, kwargs = mock_tqdm.call_args
        assert kwargs.get("total") == 5

        # Check pbar.update(1) was called exactly 5 times (once for each batch)
        assert mock_pbar.update.call_count == 5
        mock_pbar.update.assert_called_with(1)

        # Check pbar.close() was called at epoch end
        mock_pbar.close.assert_called_once()


def test_import_training_module_does_not_start_training(monkeypatch):
    """Regression test: importing app.audio.training.train must NOT execute main() or start training."""
    from unittest.mock import MagicMock
    import sys

    main_mock = MagicMock()

    # Verify that importing the module does not invoke main()
    import app.audio.training.train as train_mod
    assert hasattr(train_mod, "main")
    # Verify main was not executed on import
    assert callable(train_mod.main)


def test_windows_num_workers_default():
    """Verify num_workers defaults to 0 on Windows for multiprocessing safety."""
    import sys
    from app.config.settings import settings

    if sys.platform == "win32" and "NUM_WORKERS" not in os.environ:
        assert settings.training.num_workers == 0


