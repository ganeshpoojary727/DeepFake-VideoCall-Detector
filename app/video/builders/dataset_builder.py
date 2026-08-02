"""Video dataset builder module."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional
import torch
from torch.utils.data import Dataset

from app.video.configs.dataset_config import DatasetConfig
from app.video.registry.dataset_registry import DatasetRegistry, dataset_registry


class DatasetBuilder:
    """Builder for instantiating video PyTorch datasets from configuration."""

    def __init__(self, registry: Optional[DatasetRegistry] = None) -> None:
        self._registry = registry or dataset_registry

    def build(
        self,
        config: Optional[DatasetConfig] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        override_params: Optional[Dict[str, Any]] = None,
    ) -> Dataset:
        """Construct PyTorch video dataset instance from DatasetConfig.

        Args:
            config: Dataset configuration object.
            transform: Optional tensor transform pipeline callable.
            override_params: Optional parameter override dictionary.

        Returns:
            Dataset: Instantiated PyTorch dataset.
        """
        cfg = config or DatasetConfig()
        ds_name = cfg.dataset_name

        try:
            ds_cls = self._registry.get(ds_name)
            kwargs: Dict[str, Any] = {
                "config": cfg,
                "transform": transform,
            }
            if override_params:
                kwargs.update(override_params)
            return ds_cls(**kwargs)
        except Exception:
            from app.video.datasets.dataset_factory import DatasetFactory
            return DatasetFactory.create_dataset(cfg, transform=transform)
