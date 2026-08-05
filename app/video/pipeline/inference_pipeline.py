"""End-to-end video production inference pipeline module.

Pipeline Architecture:
Input Video -> VideoDecoder -> FrameExtractor -> FrameSampler -> FaceDetector
  -> FaceAligner -> FaceCropper -> ResolutionConverter -> VideoTensorConverter
  -> VideoNormalizer -> EfficientNet / Temporal Model -> InferencePostProcessor -> InferenceResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from app.utils.logger import get_logger
from app.video.configs.inference_config import VideoInferenceConfig
from app.video.exceptions.video_exceptions import ConfigurationError, PreprocessingError
from app.video.inference.postprocess import InferencePostProcessor
from app.video.preprocessing.face_aligner import FaceAligner
from app.video.preprocessing.face_cropper import FaceCropper
from app.video.preprocessing.face_detector import FaceDetector
from app.video.preprocessing.frame_extractor import FrameExtractor
from app.video.preprocessing.frame_sampler import FrameSampler
from app.video.preprocessing.resolution_converter import ResolutionConverter
from app.video.preprocessing.video_decoder import VideoDecoder
from app.video.preprocessing.video_normalizer import VideoNormalizer
from app.video.preprocessing.video_tensor_converter import VideoTensorConverter

logger = get_logger(__name__)


@dataclass
class InferenceResult:
    """Structured container for video deepfake inference results and timing metrics."""

    is_fake: bool
    is_deepfake: bool
    fake_probability: float
    real_probability: float
    confidence: float
    label: int
    label_name: str
    num_frames: int
    num_faces_detected: int
    preprocessing_time_ms: float
    inference_time_ms: float
    postprocessing_time_ms: float
    total_runtime_ms: float
    device: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access for backwards compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.metadata:
            return self.metadata[key]
        raise KeyError(f"Key '{key}' not found in InferenceResult.")

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for dict compatibility."""
        return hasattr(self, key) or key in self.metadata

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-like get method."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.metadata.get(key, default)

    def keys(self) -> List[str]:
        """Return all available keys."""
        d = self.to_dict()
        return list(d.keys())

    def values(self) -> List[Any]:
        """Return all available values."""
        d = self.to_dict()
        return list(d.values())

    def items(self) -> List[Tuple[str, Any]]:
        """Return items key-value pairs."""
        d = self.to_dict()
        return list(d.items())

    def to_dict(self) -> Dict[str, Any]:
        """Convert result object to standard dictionary format."""
        return {
            "is_fake": self.is_fake,
            "is_deepfake": self.is_deepfake,
            "fake_probability": self.fake_probability,
            "real_probability": self.real_probability,
            "confidence": self.confidence,
            "label": self.label,
            "label_name": self.label_name,
            "num_frames": self.num_frames,
            "num_faces_detected": self.num_faces_detected,
            "preprocessing_time_ms": self.preprocessing_time_ms,
            "inference_time_ms": self.inference_time_ms,
            "postprocessing_time_ms": self.postprocessing_time_ms,
            "total_runtime_ms": self.total_runtime_ms,
            "device": self.device,
            "metadata": self.metadata,
        }


class InferencePipeline:
    """Production video deepfake inference pipeline implementing stage-by-stage runtime orchestration."""

    def __init__(
        self,
        model: Any,
        config: Optional[VideoInferenceConfig] = None,
    ) -> None:
        """Initialize production video inference pipeline.

        Args:
            model: PyTorch neural network or generic inference runner.
            config: Optional VideoInferenceConfig instance.
        """
        self.config = config or VideoInferenceConfig()
        self.config.validate()

        # Device management: resolution with fallback
        requested_device = self.config.device.lower()
        use_cuda = "cuda" in requested_device and torch.cuda.is_available()

        if use_cuda:
            try:
                dev = torch.device(self.config.device)
                if dev.index is not None and dev.index >= torch.cuda.device_count():
                    logger.warning(
                        "CUDA device index %d out of range (count=%d). Falling back to CPU.",
                        dev.index,
                        torch.cuda.device_count(),
                    )
                    self.device = torch.device("cpu")
                else:
                    self.device = dev
            except Exception as exc:
                logger.warning("CUDA device initialization failed (%s). Falling back to CPU.", exc)
                self.device = torch.device("cpu")
        else:
            if "cuda" in requested_device:
                logger.warning("CUDA requested in config but unavailable. Falling back to CPU.")
            self.device = torch.device("cpu")

        self.model = model
        if isinstance(self.model, nn.Module):
            try:
                self.model.to(self.device)
            except Exception as exc:
                logger.warning("Moving model to device %s failed (%s). Falling back to CPU.", self.device, exc)
                self.device = torch.device("cpu")
                self.model.to(self.device)
            self.model.eval()

        logger.info("Initializing InferencePipeline on device %s", self.device)

        # Stage Orchestration Components
        self.decoder = VideoDecoder(target_fps=self.config.target_fps)
        self.frame_extractor = FrameExtractor(max_frames=self.config.sequence_length * 2)
        self.frame_sampler = FrameSampler(
            num_frames=self.config.sequence_length,
            strategy="uniform",
            stride=self.config.frame_stride,
        )
        self.face_detector = FaceDetector(conf_threshold=0.6)
        self.face_aligner = FaceAligner(output_size=self.config.target_resolution)
        self.face_cropper = FaceCropper(margin=self.config.face_margin, target_size=self.config.target_resolution)
        self.resolution_converter = ResolutionConverter(target_resolution=self.config.target_resolution)
        self.tensor_converter = VideoTensorConverter(scale_to_unit=True)
        self.normalizer = VideoNormalizer()
        self.postprocessor = InferencePostProcessor(confidence_threshold=self.config.confidence_threshold)

    def _validate_input(self, video_input: Union[str, bytes, np.ndarray, List[np.ndarray]]) -> None:
        """Validate input type and existence before preprocessing."""
        if video_input is None:
            raise PreprocessingError("Video input cannot be None")
        if isinstance(video_input, str):
            if not video_input.strip():
                raise PreprocessingError("Video file path string is empty")
            if not os.path.exists(video_input):
                raise PreprocessingError(f"Video file path does not exist: {video_input}")
        elif isinstance(video_input, bytes):
            if len(video_input) == 0:
                raise PreprocessingError("Video input bytes buffer is empty")
        elif isinstance(video_input, np.ndarray):
            if video_input.size == 0:
                raise PreprocessingError("Video input numpy array is empty")
        elif isinstance(video_input, list):
            if len(video_input) == 0:
                raise PreprocessingError("Video input frame list is empty")
        else:
            raise PreprocessingError(f"Unsupported video input type: {type(video_input)}")

    def preprocess_video(
        self, video_input: Union[str, bytes, np.ndarray, List[np.ndarray]]
    ) -> Tuple[torch.Tensor, int]:
        """Execute stage 1 to 4 preprocessing steps.

        Stages:
        1. Video Decoding & Frame Extraction
        2. Frame Sampling
        3. Face Detection, Alignment, and Cropping (with graceful fallback)
        4. Resolution Normalization & PyTorch Tensor Conversion

        Returns:
            Tuple[torch.Tensor, int]: Preprocessed 4D video tensor [T, C, H, W] and count of faces detected.
        """
        self._validate_input(video_input)

        # Stage 1: Frame Extraction & Decoding
        logger.debug("Stage 1: Decoding and extracting frames...")
        if isinstance(video_input, list):
            raw_frames = video_input
        else:
            try:
                raw_frames = self.frame_extractor.extract(video_input)
            except Exception as exc:
                logger.debug("FrameExtractor failed, falling back to VideoDecoder: %s", exc)
                raw_frames = self.decoder.decode(video_input)

        if not raw_frames:
            raise PreprocessingError("Failed to extract any frames from input video.")

        # Stage 2: Frame Sampling
        logger.debug("Stage 2: Sampling frames to sequence length %d...", self.config.sequence_length)
        sampled_frames = self.frame_sampler.sample(raw_frames)

        # Stage 3 & 4: Face Detection, Alignment, Cropping, and Spatial Normalization
        logger.debug("Stage 3: Face detection, alignment, cropping, and resolution conversion...")
        processed_crops: List[np.ndarray] = []
        faces_detected_count = 0

        for frame in sampled_frames:
            crop = None
            if self.config.crop_faces:
                try:
                    box = self.face_detector.detect_largest(frame)
                    if box is not None:
                        bbox_tuple = (box.x, box.y, box.x + box.w, box.y + box.h)
                        crop = self.face_cropper.crop(frame, bbox=bbox_tuple)
                        crop = self.face_aligner.align(crop)
                        faces_detected_count += 1
                except Exception as exc:
                    logger.warning("Face detection failed on frame: %s. Using full frame fallback.", exc)
                    crop = None

            if crop is None:
                # Center-crop / full frame fallback
                crop = self.face_cropper.crop(frame, bbox=None)
                crop = self.face_aligner.align(crop)

            # Ensure target resolution spatial conversion
            crop = self.resolution_converter.convert(crop)
            processed_crops.append(crop)

        # Convert to float PyTorch tensor [T, C, H, W]
        logger.debug("Stage 4: Tensor conversion and normalization...")
        seq_tensor = self.tensor_converter.to_tensor(processed_crops)

        if self.config.normalize:
            seq_tensor = self.normalizer.normalize(seq_tensor)

        return seq_tensor, faces_detected_count

    def _run_model(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run forward pass on model supporting PyTorch nn.Module and future ONNX/TensorRT runtimes."""
        if hasattr(self.model, "forward"):
            return self.model(input_tensor)
        elif callable(self.model):
            return self.model(input_tensor)
        elif hasattr(self.model, "run"):
            # ONNX Runtime session compatibility
            outputs = self.model.run(None, {"input": input_tensor.cpu().numpy()})
            return torch.from_numpy(outputs[0])
        else:
            raise RuntimeError(f"Unsupported model interface type: {type(self.model)}")

    def predict_video(
        self, video_input: Union[str, bytes, np.ndarray, List[np.ndarray]]
    ) -> InferenceResult:
        """Execute complete end-to-end production inference pipeline on a single video.

        Stages:
        1. Decode / Extract frames from video
        2. Sample temporal frames
        3. Detect faces, align, and crop ROI
        4. Spatial conversion & Tensor normalization
        5. Forward pass through neural network model
        6. Post-process confidence predictions

        Args:
            video_input: Video file path, raw video bytes, or frame array sequence.

        Returns:
            InferenceResult: Structured prediction result containing probabilities and runtime metrics.
        """
        start_total = time.perf_counter()

        # Stages 1 to 4: Preprocessing
        start_prep = time.perf_counter()
        seq_tensor, num_faces = self.preprocess_video(video_input)  # [T, C, H, W]
        input_tensor = seq_tensor.unsqueeze(0).to(self.device)  # [1, T, C, H, W]
        prep_time = (time.perf_counter() - start_prep) * 1000.0

        # Stage 5: Neural Model Forward Pass
        logger.debug("Stage 5: Neural network forward pass on %s...", self.device)
        start_infer = time.perf_counter()
        with torch.no_grad():
            logits = self._run_model(input_tensor)
        infer_time = (time.perf_counter() - start_infer) * 1000.0

        # Stage 6: Postprocessing
        logger.debug("Stage 6: Postprocessing confidence predictions...")
        start_post = time.perf_counter()
        processed = self.postprocessor.process_outputs(logits)
        post_time = (time.perf_counter() - start_post) * 1000.0

        total_time = (time.perf_counter() - start_total) * 1000.0

        result = InferenceResult(
            is_fake=processed["is_fake"],
            is_deepfake=processed["is_deepfake"],
            fake_probability=processed["fake_probability"],
            real_probability=processed["real_probability"],
            confidence=processed["confidence"],
            label=processed["label"],
            label_name=processed["label_name"],
            num_frames=seq_tensor.shape[0],
            num_faces_detected=num_faces,
            preprocessing_time_ms=round(prep_time, 2),
            inference_time_ms=round(infer_time, 2),
            postprocessing_time_ms=round(post_time, 2),
            total_runtime_ms=round(total_time, 2),
            device=str(self.device),
            metadata={
                "sequence_shape": list(input_tensor.shape),
                "confidence_threshold": self.config.confidence_threshold,
            },
        )

        logger.info(
            "Inference complete: label=%s, fake_prob=%.4f, prep=%.1fms, infer=%.1fms, total=%.1fms",
            result.label_name,
            result.fake_probability,
            result.preprocessing_time_ms,
            result.inference_time_ms,
            result.total_runtime_ms,
        )

        return result

    def predict_batch(
        self,
        video_inputs: List[Union[str, bytes, np.ndarray, List[np.ndarray]]],
        batch_size: Optional[int] = None,
    ) -> List[InferenceResult]:
        """Execute batched inference on multiple video inputs.

        Args:
            video_inputs: List of video file paths, bytes, or array sequences.
            batch_size: Optional batch size override. Defaults to config.batch_size.

        Returns:
            List[InferenceResult]: List of structured prediction results.
        """
        if not video_inputs:
            return []

        bs = batch_size or self.config.batch_size
        results: List[InferenceResult] = []

        for i in range(0, len(video_inputs), bs):
            chunk = video_inputs[i : i + bs]
            tensors = []
            num_faces_list = []

            start_prep = time.perf_counter()
            for v_in in chunk:
                seq_tensor, num_faces = self.preprocess_video(v_in)
                tensors.append(seq_tensor)
                num_faces_list.append(num_faces)
            prep_time = (time.perf_counter() - start_prep) * 1000.0 / len(chunk)

            # Collate into 5D batch tensor [B, T, C, H, W]
            batch_tensor = torch.stack(tensors, dim=0).to(self.device)

            start_infer = time.perf_counter()
            with torch.no_grad():
                logits = self._run_model(batch_tensor)
            infer_time = (time.perf_counter() - start_infer) * 1000.0 / len(chunk)

            for idx in range(len(chunk)):
                start_post = time.perf_counter()
                item_logits = logits[idx : idx + 1]
                processed = self.postprocessor.process_outputs(item_logits)
                post_time = (time.perf_counter() - start_post) * 1000.0

                res = InferenceResult(
                    is_fake=processed["is_fake"],
                    is_deepfake=processed["is_deepfake"],
                    fake_probability=processed["fake_probability"],
                    real_probability=processed["real_probability"],
                    confidence=processed["confidence"],
                    label=processed["label"],
                    label_name=processed["label_name"],
                    num_frames=tensors[idx].shape[0],
                    num_faces_detected=num_faces_list[idx],
                    preprocessing_time_ms=round(prep_time, 2),
                    inference_time_ms=round(infer_time, 2),
                    postprocessing_time_ms=round(post_time, 2),
                    total_runtime_ms=round(prep_time + infer_time + post_time, 2),
                    device=str(self.device),
                    metadata={
                        "sequence_shape": list(tensors[idx].shape),
                        "confidence_threshold": self.config.confidence_threshold,
                        "batch_index": idx,
                    },
                )
                results.append(res)

        return results
