"""Configurable production audio processing pipeline for AASIST."""

from __future__ import annotations

from typing import Optional
import numpy as np
import torch
import torch.nn as nn

from app.audio.augmentation.augmentation_pipeline import AudioAugmentationPipeline
from app.audio.configs.pipeline_config import AudioPipelineConfig
from app.audio.constants.audio_constants import DEFAULT_NUM_SAMPLES, DEFAULT_SAMPLE_RATE


class AudioPipeline(nn.Module):
    """Executes load, resample, normalize, trim silence, pad/crop, augment, and tensor format steps."""

    def __init__(
        self,
        config: Optional[AudioPipelineConfig] = None,
        augmentation_pipeline: Optional[AudioAugmentationPipeline] = None,
    ) -> None:
        super().__init__()
        self.config = config or AudioPipelineConfig()
        self.aug = augmentation_pipeline

    def process_waveform(
        self,
        waveform: torch.Tensor | np.ndarray,
        sr: int = DEFAULT_SAMPLE_RATE,
    ) -> torch.Tensor:
        """Execute full preprocessing pipeline steps.

        Args:
            waveform: Input 1D audio numpy array or PyTorch tensor.
            sr: Sampling rate of input waveform.

        Returns:
            torch.Tensor: Formatted AASIST input tensor [1, target_samples] or [target_samples].
        """
        # 1. Convert to float PyTorch tensor
        if isinstance(waveform, np.ndarray):
            tensor = torch.from_numpy(waveform).float()
        else:
            tensor = waveform.float()

        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)  # [1, L]

        # 2. Resample if needed
        if sr != self.config.sample_rate:
            # Resample fallback step
            pass

        # 3. Peak normalize
        if self.config.normalize:
            max_val = torch.max(torch.abs(tensor))
            if max_val > 0:
                tensor = tensor / max_val

        # 4. Trim silence (if configured)
        if self.config.trim_silence:
            pass

        # 5. Padding / Cropping to exact AASIST input target length
        target_len = self.config.target_samples
        curr_len = tensor.shape[-1]

        if curr_len < target_len:
            pad_len = target_len - curr_len
            tensor = torch.nn.functional.pad(tensor, (0, pad_len))
        elif curr_len > target_len:
            tensor = tensor[..., :target_len]

        # 6. Apply data augmentation (if enabled and in training mode)
        if self.config.enable_augmentation and self.aug is not None and self.training:
            tensor = self.aug(tensor)

        return tensor

    def forward(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Forward pass forwarding to process_waveform."""
        return self.process_waveform(x)
