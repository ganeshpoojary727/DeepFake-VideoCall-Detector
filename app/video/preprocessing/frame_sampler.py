"""Temporal frame sampling module."""

from __future__ import annotations

from typing import List
import numpy as np

from app.video.exceptions.video_exceptions import PreprocessingError


class FrameSampler:
    """Samples subset frame indices from video sequence using configurable strategies."""

    def __init__(
        self,
        num_frames: int = 16,
        strategy: str = "uniform",
        stride: int = 1,
    ) -> None:
        self._num_frames = num_frames
        self._strategy = strategy.lower().strip()
        self._stride = stride

    def sample(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Sample frame list using specified sampling strategy.

        Args:
            frames: Source list of frame arrays.

        Returns:
            List[np.ndarray]: Sampled subset list of frame arrays.

        Raises:
            PreprocessingError: If frame list is empty or strategy is unknown.
        """
        if not frames:
            raise PreprocessingError("Cannot sample from empty frame list.")

        total = len(frames)
        target = self._num_frames

        if self._strategy == "uniform":
            indices = np.linspace(0, total - 1, target, dtype=int)
        elif self._strategy == "stride":
            indices = list(range(0, total, self._stride))[:target]
            if len(indices) < target:
                last = indices[-1] if indices else 0
                indices += [last] * (target - len(indices))
        elif self._strategy == "random":
            if total >= target:
                indices = np.sort(np.random.choice(total, target, replace=False))
            else:
                indices = np.random.choice(total, target, replace=True)
        elif self._strategy == "center":
            mid = total // 2
            start = max(0, mid - target // 2)
            indices = list(range(start, min(total, start + target)))
            if len(indices) < target:
                indices += [indices[-1]] * (target - len(indices))
        else:
            raise PreprocessingError(f"Unknown sampling strategy '{self._strategy}'")

        return [frames[i] for i in indices]
