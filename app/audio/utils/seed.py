"""
Reproducibility and seed management utility for pseudo-random generators.

Provides the SeedManager class for controlling random seeds across Python random,
NumPy, PyTorch CPU, and CUDA execution backends.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

from app.config.settings import settings


class SeedManager:
    """
    Manager for setting random seeds to guarantee experiment reproducibility.

    Parameters
    ----------
    seed : Optional[int]
        Random seed value. Defaults to settings.training.seed if None.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed = seed if seed is not None else settings.training.seed

    @property
    def seed(self) -> int:
        """Get currently configured random seed."""
        return self._seed

    def set_seed(self, deterministic: bool = True) -> int:
        """
        Apply random seed across all libraries and backend frameworks.

        Parameters
        ----------
        deterministic : bool
            If True, configures CuDNN backends to execute deterministically.

        Returns
        -------
        int
            The applied random seed value.
        """
        os.environ["PYTHONHASHSEED"] = str(self._seed)
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self._seed)
            torch.cuda.manual_seed_all(self._seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            else:
                torch.backends.cudnn.benchmark = True

        return self._seed

    @staticmethod
    def get_generator(seed: Optional[int] = None) -> torch.Generator:
        """
        Create a seeded PyTorch Generator for DataLoaders and sampling.

        Parameters
        ----------
        seed : Optional[int]
            Seed for the generator. If None, uses settings.training.seed.

        Returns
        -------
        torch.Generator
            Seeded PyTorch random generator instance.
        """
        effective_seed = seed if seed is not None else settings.training.seed
        generator = torch.Generator()
        generator.manual_seed(effective_seed)
        return generator
