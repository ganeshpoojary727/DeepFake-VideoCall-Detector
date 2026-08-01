"""
Video preprocessing pipeline for deepfake face detection.

Pipeline
--------
1. Frame extraction from video file (or single frame input)
2. Face detection via FaceDetector (Haar cascade)
3. Face crop + resize to 224×224
4. Normalization (ImageNet mean/std)
5. Optional augmentation (flip, brightness, blur, compression artifacts)
6. Return as torch.Tensor of shape (C, H, W)

Supports both single-frame and multi-frame (video file) inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np
import torch

from app.video.face_detection.face_detector import FaceBox, FaceDetector
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ImageNet normalization constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FACE_SIZE = (224, 224)


class VideoPreprocessor:
    """
    Preprocessing pipeline for video frames.

    Parameters
    ----------
    face_size : tuple[int, int]
        (width, height) of the output face crop.
    max_frames : int
        Maximum number of frames to extract from a video file.
    frame_stride : int
        Extract every N-th frame to reduce redundancy.
    apply_augmentation : bool
        Whether to apply random augmentation (for training only).
    """

    def __init__(
        self,
        face_size: Tuple[int, int] = FACE_SIZE,
        max_frames: int = 16,
        frame_stride: int = 4,
        apply_augmentation: bool = False,
    ) -> None:
        self.face_size = face_size
        self.max_frames = max_frames
        self.frame_stride = frame_stride
        self.apply_augmentation = apply_augmentation
        self._detector = FaceDetector()

    # ── Public API ────────────────────────────

    def process_frame(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        """
        Process a single BGR frame.

        Returns a face tensor of shape ``(3, H, W)`` in float32,
        or ``None`` if no face is detected.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H, W, 3).
        """
        box = self._detector.detect_largest(frame)
        if box is None:
            return None

        face_crop = self._detector.crop_face(frame, box, target_size=self.face_size)
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        if self.apply_augmentation:
            face_rgb = self._augment(face_rgb)

        return self._to_tensor(face_rgb)

    def process_video(
        self, video_path: str | Path
    ) -> List[torch.Tensor]:
        """
        Extract face tensors from a video file.

        Parameters
        ----------
        video_path : str | Path
            Path to the video file (.mp4, .avi, .mov, etc.).

        Returns
        -------
        List[torch.Tensor]
            List of face tensors of shape ``(3, H, W)``.
            Empty list if no faces found or file unreadable.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Cannot open video: %s", video_path)
            return []

        tensors: List[torch.Tensor] = []
        frame_idx = 0

        try:
            while len(tensors) < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.frame_stride == 0:
                    tensor = self.process_frame(frame)
                    if tensor is not None:
                        tensors.append(tensor)

                frame_idx += 1
        finally:
            cap.release()

        logger.debug(
            "Extracted %d face frames from %s (total: %d frames read)",
            len(tensors), video_path.name, frame_idx,
        )
        return tensors

    def frames_from_video(
        self, video_path: str | Path
    ) -> Iterator[np.ndarray]:
        """
        Yield BGR frames from a video file at ``frame_stride`` intervals.

        Useful for streaming processing without loading all frames into memory.
        """
        cap = cv2.VideoCapture(str(video_path))
        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % self.frame_stride == 0:
                    yield frame
                idx += 1
        finally:
            cap.release()

    # ── Private helpers ───────────────────────

    @staticmethod
    def _to_tensor(rgb_image: np.ndarray) -> torch.Tensor:
        """
        Convert an RGB uint8 image to a normalised float32 tensor.

        Returns shape ``(3, H, W)`` with ImageNet normalisation.
        """
        img = rgb_image.astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        return torch.from_numpy(img.transpose(2, 0, 1))  # (C, H, W)

    @staticmethod
    def _augment(image: np.ndarray) -> np.ndarray:
        """Apply random augmentation to a uint8 RGB image."""
        # Random horizontal flip
        if np.random.random() < 0.5:
            image = cv2.flip(image, 1)

        # Random brightness/contrast
        alpha = np.random.uniform(0.8, 1.2)
        beta = np.random.randint(-20, 20)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        # Random Gaussian blur (simulates compression artifacts)
        if np.random.random() < 0.3:
            ksize = np.random.choice([3, 5])
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)

        # JPEG compression simulation
        if np.random.random() < 0.3:
            quality = np.random.randint(50, 95)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, enc = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), encode_param)
            image = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        return image
