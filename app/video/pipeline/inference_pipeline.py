"""End-to-end video inference pipeline module."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from app.video.configs.inference_config import VideoInferenceConfig
from app.video.preprocessing.frame_extractor import FrameExtractor
from app.video.preprocessing.frame_sampler import FrameSampler
from app.video.preprocessing.resolution_converter import ResolutionConverter
from app.video.preprocessing.sequence_builder import SequenceBuilder
from app.video.preprocessing.video_normalizer import VideoNormalizer
from app.video.preprocessing.video_tensor_converter import VideoTensorConverter
from app.video.utils.device import get_device


class InferencePipeline:
    """Executes frame extraction, preprocessing, sequence building, and deepfake detection inference."""

    def __init__(
        self,
        model: nn.Module,
        config: Optional[VideoInferenceConfig] = None,
    ) -> None:
        self.config = config or VideoInferenceConfig()
        self.model = model
        self.device = get_device(self.config.device)
        self.model.to(self.device)
        self.model.eval()

        self._extractor = FrameExtractor()
        self._sampler = FrameSampler(
            num_frames=self.config.sequence_length,
            stride=self.config.frame_stride,
        )
        self._resizer = ResolutionConverter(target_resolution=self.config.target_resolution)
        self._builder = SequenceBuilder(sequence_length=self.config.sequence_length)
        self._converter = VideoTensorConverter()
        self._normalizer = VideoNormalizer()

    def predict_video(self, video_input: str | np.ndarray) -> Dict[str, Any]:
        """Run deepfake prediction on input video path or numpy array sequence.

        Args:
            video_input: Video file path string or frame sequence numpy array.

        Returns:
            Dict[str, Any]: Prediction payload with probabilities and binary classification label.
        """
        frames = self._extractor.extract(video_input)
        sampled = self._sampler.sample(frames)
        resized = self._resizer.convert_batch(sampled)
        seq = self._builder.build(resized)
        tensor = self._converter.to_tensor(seq)  # [T, C, H, W]

        if self.config.normalize:
            tensor = self._normalizer.normalize(tensor)

        tensor = tensor.unsqueeze(0).to(self.device)  # [1, T, C, H, W]

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        fake_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
        is_fake = fake_prob >= self.config.confidence_threshold

        return {
            "is_fake": bool(is_fake),
            "fake_probability": fake_prob,
            "real_probability": 1.0 - fake_prob,
            "logits": logits.squeeze(0).cpu().numpy().tolist(),
        }
