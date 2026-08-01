"""Video frame sequence builder module."""

from __future__ import annotations

from typing import List
import numpy as np

from app.video.exceptions.video_exceptions import PreprocessingError


class SequenceBuilder:
    """Assembles individual frame images into contiguous video sequence arrays."""

    def __init__(self, sequence_length: int = 16, pad_if_short: bool = True) -> None:
        self._sequence_length = sequence_length
        self._pad_if_short = pad_if_short

    def build(self, frames: List[np.ndarray]) -> np.ndarray:
        """Pack list of [H, W, C] frame arrays into [T, H, W, C] sequence tensor array.

        Args:
            frames: List of 3D frame arrays.

        Returns:
            np.ndarray: Stacked sequence array [T, H, W, C].

        Raises:
            PreprocessingError: If input frame list is empty.
        """
        if not frames:
            raise PreprocessingError("Cannot build sequence from empty frame list.")

        n = len(frames)
        target_len = self._sequence_length

        if n >= target_len:
            selected = frames[:target_len]
        else:
            if self._pad_if_short:
                last_frame = frames[-1]
                needed = target_len - n
                selected = frames + [last_frame.copy() for _ in range(needed)]
            else:
                selected = frames

        sequence = np.stack(selected, axis=0)
        return sequence
