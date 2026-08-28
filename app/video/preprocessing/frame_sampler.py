"""
Temporal Frame Sampling Module for Video Deepfake Forensics.

Supports uniform, fixed-stride, random, and center frame sampling strategies
with frame index tracking and timestamp calculation for temporal timeline telemetry.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from app.video.exceptions.video_exceptions import PreprocessingError


class FrameSampler:
    """Samples subset frame sequences from video streams with temporal metadata."""

    def __init__(
        self,
        num_frames: int = 16,
        strategy: str = "uniform",
        stride: int = 1,
    ) -> None:
        self._num_frames = num_frames
        self._strategy = strategy.lower().strip()
        self._stride = max(1, stride)

    @property
    def num_frames(self) -> int:
        """Target sequence length."""
        return self._num_frames

    @property
    def strategy(self) -> str:
        """Sampling strategy name."""
        return self._strategy

    def get_sample_indices(self, total_frames: int) -> List[int]:
        """Compute sampled frame indices given total frame count in video."""
        if total_frames <= 0:
            return []

        target = self._num_frames

        if self._strategy == "uniform":
            if total_frames >= target:
                indices = np.linspace(0, total_frames - 1, target, dtype=int).tolist()
            else:
                # Repeat frames if video has fewer frames than required sequence length
                repeat_times = int(np.ceil(target / total_frames))
                extended = (list(range(total_frames)) * repeat_times)[:target]
                indices = extended

        elif self._strategy == "stride":
            indices = list(range(0, total_frames, self._stride))[:target]
            if len(indices) < target:
                last_idx = indices[-1] if indices else 0
                indices += [last_idx] * (target - len(indices))

        elif self._strategy == "random":
            if total_frames >= target:
                indices = np.sort(np.random.choice(total_frames, target, replace=False)).tolist()
            else:
                indices = np.random.choice(total_frames, target, replace=True).tolist()

        elif self._strategy == "center":
            mid = total_frames // 2
            start = max(0, mid - target // 2)
            indices = list(range(start, min(total_frames, start + target)))
            if len(indices) < target:
                last_idx = indices[-1] if indices else 0
                indices += [last_idx] * (target - len(indices))

        else:
            raise PreprocessingError(f"Unknown sampling strategy '{self._strategy}'")

        return indices

    def sample(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Sample frame list using configured sampling strategy.

        Args:
            frames: Source list of frame arrays.

        Returns:
            List[np.ndarray]: Sampled subset list of frame arrays.
        """
        if not frames:
            raise PreprocessingError("Cannot sample from empty frame list.")

        indices = self.get_sample_indices(len(frames))
        return [frames[i] for i in indices]

    def sample_with_metadata(
        self,
        frames: List[np.ndarray],
        fps: float = 30.0,
    ) -> List[Tuple[np.ndarray, int, float]]:
        """Sample frames while tracking original frame index and timestamp in seconds.

        Args:
            frames: Source list of frame arrays.
            fps: Video frames per second (default: 30.0).

        Returns:
            List[Tuple[np.ndarray, int, float]]:
                List of (frame_array, original_frame_idx, timestamp_sec).
        """
        if not frames:
            raise PreprocessingError("Cannot sample from empty frame list.")

        safe_fps = fps if fps > 0 else 30.0
        indices = self.get_sample_indices(len(frames))

        results: List[Tuple[np.ndarray, int, float]] = []
        for idx in indices:
            frame = frames[idx]
            timestamp_sec = float(idx / safe_fps)
            results.append((frame, idx, round(timestamp_sec, 3)))

        return results
