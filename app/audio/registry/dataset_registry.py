"""Dataset registry module.

Provides DatasetRegistry for registering, looking up, and instantiating audio
datasets (ASVspoof2019, WaveFake, InTheWild, etc.).
"""

from __future__ import annotations

from torch.utils.data import Dataset

from app.audio.registry.base_registry import BaseRegistry


class DatasetRegistry(BaseRegistry[Dataset]):
    """Registry for audio dataset implementations."""

    def __init__(self) -> None:
        super().__init__(name="DatasetRegistry")


# Default global instance for dataset classes
dataset_registry = DatasetRegistry()
