"""Color, brightness, and contrast video augmentations."""

from __future__ import annotations

import torch

from app.video.augmentation.base_augmentation import BaseVideoAugmentation


class Brightness(BaseVideoAugmentation):
    """Adjusts video frame brightness uniformly."""

    def __init__(self, factor: float = 0.2, p: float = 0.5) -> None:
        super().__init__(p=p)
        self._factor = factor

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        """Apply random brightness shift."""
        shift = (torch.rand(1).item() * 2.0 - 1.0) * self._factor
        return torch.clamp(video + shift, 0.0, 1.0)


class Contrast(BaseVideoAugmentation):
    """Adjusts video frame contrast around mean luminance."""

    def __init__(self, factor: float = 0.2, p: float = 0.5) -> None:
        super().__init__(p=p)
        self._factor = factor

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        """Apply contrast scaling."""
        scale = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * self._factor
        mean = video.mean(dim=(-2, -1), keepdim=True)
        return torch.clamp((video - mean) * scale + mean, 0.0, 1.0)


class ColorJitter(BaseVideoAugmentation):
    """Applies combined brightness and contrast color jittering."""

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, p: float = 0.5) -> None:
        super().__init__(p=p)
        self._b_aug = Brightness(factor=brightness, p=1.0)
        self._c_aug = Contrast(factor=contrast, p=1.0)

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        """Apply brightness and contrast transforms sequentially."""
        v = self._b_aug.apply(video)
        return self._c_aug.apply(v)
