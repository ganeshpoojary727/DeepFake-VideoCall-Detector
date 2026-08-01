"""Video visualization and frame bounding box annotation utility functions."""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import torch


def draw_bboxes(
    frame: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw facial bounding boxes onto frame array (H, W, C).

    Args:
        frame: Target image frame array (H, W, 3).
        bboxes: Bounding box coordinates list [(x1, y1, x2, y2)].
        color: RGB line color tuple.
        thickness: Line stroke width.

    Returns:
        np.ndarray: Frame array with annotated bounding box rectangles.
    """
    annotated = frame.copy()
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
        annotated[y1 : y1 + thickness, x1:x2] = color
        annotated[y2 - thickness : y2, x1:x2] = color
        annotated[y1:y2, x1 : x1 + thickness] = color
        annotated[y1:y2, x2 - thickness : x2] = color
    return annotated


def visualize_frames(frames: torch.Tensor | np.ndarray, max_frames: int = 8) -> np.ndarray:
    """Grid tile sequence of video frames into a single image strip array.

    Args:
        frames: Video frames tensor [T, C, H, W] or array [T, H, W, C].
        max_frames: Max frames to tile.

    Returns:
        np.ndarray: Combined horizontal frame strip array.
    """
    if isinstance(frames, torch.Tensor):
        if frames.dim() == 4 and frames.shape[1] in (1, 3):
            arr = frames.permute(0, 2, 3, 1).cpu().numpy()
        else:
            arr = frames.cpu().numpy()
    else:
        arr = frames

    n = min(len(arr), max_frames)
    selected = arr[:n]
    strip = np.concatenate(selected, axis=1)
    return strip


def plot_training_curves(
    train_losses: List[float], val_losses: List[float]
) -> dict[str, List[float]]:
    """Return dictionary structure of training history metrics for plotting.

    Args:
        train_losses: List of training loss values.
        val_losses: List of validation loss values.

    Returns:
        dict[str, List[float]]: Formatted history dict.
    """
    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
    }
