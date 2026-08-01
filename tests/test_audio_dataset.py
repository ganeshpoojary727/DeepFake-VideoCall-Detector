"""Tests for the AudioDataset."""

from __future__ import annotations

import pytest
from pathlib import Path

from app.audio.datasets.audio_dataset import AudioDataset


class TestAudioDataset:
    """Test suite for AudioDataset."""

    def test_missing_protocol_raises(self) -> None:
        """AudioDataset should raise FileNotFoundError for missing protocol."""
        with pytest.raises(FileNotFoundError):
            AudioDataset(
                protocol_file="/nonexistent/protocol.txt",
                audio_directory="/nonexistent/audio",
            )

    def test_constructor_requires_args(self) -> None:
        """AudioDataset requires protocol_file and audio_directory."""
        with pytest.raises(TypeError):
            AudioDataset()  # type: ignore[call-arg]