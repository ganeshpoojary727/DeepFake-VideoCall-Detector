"""Random seed manager utility module for reproducibility."""

from __future__ import annotations

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed across random, numpy, and torch for reproducible results.

    Args:
        seed: Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SeedManager:
    """Stateful random seed management class."""

    def __init__(self, default_seed: int = 42) -> None:
        self._current_seed = default_seed
        self.apply(default_seed)

    @property
    def current_seed(self) -> int:
        """Get current seed value."""
        return self._current_seed

    def apply(self, seed: int) -> None:
        """Apply new seed."""
        self._current_seed = seed
        set_seed(seed)
