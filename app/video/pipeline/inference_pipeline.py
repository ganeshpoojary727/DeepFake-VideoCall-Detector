"""End-to-end video production inference pipeline module.

Pipeline Architecture:
Video -> Frame Extraction -> Face Detection -> Face Alignment -> Preprocessing -> Feature Extraction -> Model -> Prediction
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn

from app.video.configs.inference_config import VideoInferenceConfig
from app.video.inference.postprocess import InferencePostProcessor
from app.video.preprocessing.face_aligner import FaceAligner
from app.video.preprocessing.face_cropper import FaceCropper
from app.video.preprocessing.face_detector import FaceDetector
from app.video.preprocessing.frame_extractor import FrameExtractor
from app.video.preprocessing.frame_sampler import FrameSampler
from app.video.preprocessing.video_decoder import VideoDecoder
from app.video.preprocessing.video_normalizer import VideoNormalizer


class InferencePipeline:
    """Production video deepfake inference pipeline implementing stage-by-stage architecture."""

    def __init__(
        self,
        model: nn.Module,
        config: Optional[VideoInferenceConfig] = None,
    ) -> None:
        self.config = config or VideoInferenceConfig()
        self.model = model
        self.device = torch.device(self.config.device if torch.cuda.is_available() and "cuda" in self.config.device else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Modular Pipeline Stages
        self.decoder = VideoDecoder()
        self.frame_extractor = FrameExtractor(max_frames=self.config.sequence_length * 2)
        self.face_detector = FaceDetector()
        self.face_aligner = FaceAligner(output_size=self.config.target_resolution)
        self.face_cropper = FaceCropper(margin=self.config.face_margin, target_size=self.config.target_resolution)
        self.frame_sampler = FrameSampler(num_frames=self.config.sequence_length, stride=self.config.frame_stride)
        self.normalizer = VideoNormalizer()
        self.postprocessor = InferencePostProcessor(confidence_threshold=self.config.confidence_threshold)

    def predict_video(self, video_input: str | bytes | np.ndarray) -> Dict[str, Any]:
        """Execute end-to-end production inference pipeline.

        Stages:
        1. Decode / Extract frames from video
        2. Detect faces in frames
        3. Align & Crop detected facial regions
        4. Preprocess & Normalize frame tensors
        5. Forward pass through neural network model
        6. Post-process confidence predictions

        Args:
            video_input: Video file path, raw video bytes, or frame array sequence.

        Returns:
            Dict[str, Any]: Structured prediction payload dictionary.
        """
        # Stage 1: Frame Extraction & Decoding
        raw_frames = self.frame_extractor.extract(video_input)
        if not raw_frames:
            raw_frames = self.decoder.decode(video_input)

        # Stage 2 & 3: Face Detection, Alignment, and Cropping
        processed_crops: List[np.ndarray] = []
        for frame in raw_frames:
            box = self.face_detector.detect_largest(frame)
            if box is not None:
                bbox_tuple = (box.x, box.y, box.x + box.w, box.y + box.h)
                crop = self.face_cropper.crop(frame, bbox=bbox_tuple)
                crop = self.face_aligner.align(crop)
            else:
                crop = self.face_cropper.crop(frame, bbox=None)
            processed_crops.append(crop)

        # Stage 4: Temporal Sampling & Preprocessing
        sampled_crops = self.frame_sampler.sample(processed_crops)
        
        # Convert crops to [T, C, H, W] PyTorch tensor
        tensor_list = [torch.from_numpy(c).permute(2, 0, 1).float() / 255.0 for c in sampled_crops]
        seq_tensor = torch.stack(tensor_list, dim=0)  # [T, C, H, W]

        if self.config.normalize:
            seq_tensor = self.normalizer.normalize(seq_tensor)

        input_tensor = seq_tensor.unsqueeze(0).to(self.device)  # [1, T, C, H, W]

        # Stage 5 & 6: Model Forward Pass & Prediction Post-processing
        with torch.no_grad():
            logits = self.model(input_tensor)

        result = self.postprocessor.process_outputs(logits)
        result["num_frames"] = len(sampled_crops)
        return result
