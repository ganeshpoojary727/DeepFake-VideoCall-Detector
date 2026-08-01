"""Tests for the DataLoader factory functions."""

from __future__ import annotations

import torch

from app.audio.datasets.dataloader import compute_class_weights


class TestDataLoaderFactory:
    """Test suite for DataLoader utilities."""

    def test_class_weights_shape(self) -> None:
        """Class weights should be a tensor of shape (2,)."""
        weights = compute_class_weights()
        assert weights.shape == (2,)

    def test_class_weights_bonafide_higher(self) -> None:
        """Bonafide weight should be higher (minority class)."""
        weights = compute_class_weights(num_bonafide=2580, num_spoof=22800)
        assert weights[0] > weights[1], "Bonafide should have higher weight"

    def test_class_weights_dtype(self) -> None:
        """Class weights should be float32."""
        weights = compute_class_weights()
        assert weights.dtype == torch.float32