"""
Video Deepfake Detector — Multi-Signal Frame Analysis, Grad-CAM Saliency, & Telemetry.

Performs face detection, temporal frame sampling, classical forensics (ELA, 2D FFT,
Laplacian boundary consistency), neural model inference, and returns structured
forensic telemetry (verdict, confidence, raw scores, visual cues, timeline, and key artifacts).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from app.analyzer.image_forensics import ImageForensics
from app.config.settings import settings
from app.utils.logger import get_logger
from app.video.models.efficientnet.model import EfficientNetB4Model
from app.video.preprocessing.face_cropper import FaceCropper
from app.video.preprocessing.face_detector import FaceDetector
from app.video.preprocessing.frame_sampler import FrameSampler
from app.video.preprocessing.video_decoder import VideoDecoder
from app.video.preprocessing.video_normalizer import VideoNormalizer
from app.video.preprocessing.video_tensor_converter import VideoTensorConverter
from app.video.utils.visualization import GradCAM

logger = get_logger(__name__)


class VideoDeepfakeDetector:
    """Production Video Deepfake Detector providing multi-signal telemetry & visual explainability."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        sequence_length: int = 16,
    ) -> None:
        self.device = device or settings.DEVICE
        self.sequence_length = sequence_length

        self.sampler = FrameSampler(num_frames=sequence_length, strategy="uniform")
        self.detector = FaceDetector(conf_threshold=0.6)
        self.cropper = FaceCropper(margin=0.2, target_size=(224, 224))
        self.decoder = VideoDecoder()
        self.tensor_converter = VideoTensorConverter(scale_to_unit=True)
        self.normalizer = VideoNormalizer()

        # Neural backbone
        self.model: Optional[EfficientNetB4Model] = None
        self.gradcam: Optional[GradCAM] = None
        self._init_model(model_path)

    def _init_model(self, model_path: Optional[Union[str, Path]]) -> None:
        """Initialize EfficientNet model and GradCAM visualizer."""
        try:
            self.model = EfficientNetB4Model()
            if model_path:
                weights = Path(model_path)
            else:
                video_dir = settings.project_root / "trained_models" / "video"
                for cand in ["best_accuracy.pt", "best_model.pt", "best_auc.pt"]:
                    if (video_dir / cand).exists():
                        weights = video_dir / cand
                        break
                else:
                    weights = video_dir / "best_model.pt"

            if weights.exists():
                self.model.load_weights(str(weights), strict=False)
                logger.info("VideoDeepfakeDetector: Loaded weights from %s", weights)
            self.model.set_mode("inference")
            self.model.to(self.device)
            self.model.eval()

            self.gradcam = GradCAM(self.model)
        except Exception as exc:
            logger.warning("VideoDeepfakeDetector: Neural model initialization fallback: %s", exc)

    # ── Inference APIs ─────────────────────────────────────────────────────────

    def predict_from_frames(self, frames_list: List[np.ndarray], fps: float = 30.0) -> float:
        """Legacy buffer-based inference returning single fake probability in [0.0, 1.0]."""
        if not frames_list or len(frames_list) < 2:
            return 0.5

        detailed = self.predict_detailed(frames_list, fps=fps)
        return float(detailed["raw_scores"]["fake_prob"])

    def predict_video(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """Perform offline deepfake analysis on an uploaded video file."""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        frames = self.decoder.decode(path)
        if not frames:
            raise ValueError(f"No readable video frames found in {path}")

        # Estimate FPS via cv2 if possible
        cap = cv2.VideoCapture(str(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        cap.release()

        return self.predict_detailed(frames, fps=fps)

    def predict_detailed(
        self,
        frames_list: List[np.ndarray],
        fps: float = 30.0,
    ) -> Dict[str, Any]:
        """Perform full multi-signal analysis, per-frame scoring, and Grad-CAM saliency.

        Returns
        -------
        Dict[str, Any]
            Structured telemetry matching Phase 1 & 2 schema:
            - verdict: "REAL" | "FAKE"
            - confidence: float
            - raw_scores: {"real_prob": float, "fake_prob": float}
            - visual_cues: {"ela_discrepancy_score": float, "fft_spectral_anomaly": float, "boundary_inconsistency": float}
            - timeline: list of per-frame predictions
            - key_artifacts: list of top anomalous frames with bounding box and Grad-CAM data
        """
        if not frames_list:
            return self._fallback_result()

        safe_fps = fps if fps > 0 else 30.0
        sampled_with_meta = self.sampler.sample_with_metadata(frames_list, fps=safe_fps)

        timeline: List[Dict[str, Any]] = []
        frame_crops: List[np.ndarray] = []
        frame_bboxes: List[Optional[Tuple[int, int, int, int]]] = []
        cues_list: List[Dict[str, Any]] = []

        # 1. Per-Frame Preprocessing & Classical Forensics
        for frame, frame_idx, timestamp_sec in sampled_with_meta:
            bgr = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            face_box = self.detector.detect_largest(bgr)

            if face_box is not None:
                bbox_coords = (face_box.x, face_box.y, face_box.x + face_box.w, face_box.y + face_box.h)
                crop_bgr, actual_box = self.cropper.crop_with_bbox_metadata(bgr, bbox=bbox_coords, target_size=(224, 224))
            else:
                crop_bgr, actual_box = self.cropper.crop_with_bbox_metadata(bgr, bbox=None, target_size=(224, 224))

            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            frame_crops.append(crop_rgb)
            frame_bboxes.append(actual_box)

            # Classical forensic cues on face crop
            cues = ImageForensics.extract_visual_cues(crop_bgr, bbox=actual_box)
            cues_list.append(cues)

        # 2. Average Visual Cues Across Sequence
        avg_ela = float(np.mean([c["ela_discrepancy_score"] for c in cues_list]))
        avg_fft = float(np.mean([c["fft_spectral_anomaly"] for c in cues_list]))
        avg_boundary = float(np.mean([c["boundary_inconsistency"] for c in cues_list]))
        forensic_avg = float(np.mean([c["combined_score"] for c in cues_list]))

        # 3. Neural Sequence Forward Pass
        neural_probs: List[float] = []
        if self.model is not None and frame_crops:
            try:
                # VideoTensorConverter converts list of (224, 224, 3) frames to [1, T, C, H, W]
                unnorm = self.tensor_converter.to_tensor(frame_crops)
                tensor = self.normalizer.normalize(unnorm).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits = self.model(tensor)
                    probs = F.softmax(logits, dim=-1)
                    seq_fake_prob = float(probs[0, 1].item())

                # Per-frame neural estimation (combining spatial feature with forensic cues)
                for idx, c in enumerate(cues_list):
                    frame_p = 0.85 * seq_fake_prob + 0.15 * c["combined_score"]
                    neural_probs.append(float(np.clip(frame_p, 0.01, 0.99)))
            except Exception as exc:
                logger.debug("Neural forward pass exception: %s", exc)
                neural_probs = [float(c["combined_score"]) for c in cues_list]
        else:
            neural_probs = [float(c["combined_score"]) for c in cues_list]

        # 4. Build Timeline
        for idx, (frame, f_idx, t_sec) in enumerate(sampled_with_meta):
            f_prob = neural_probs[idx]
            is_anomaly = bool(f_prob >= 0.55)
            timeline.append({
                "frame_idx": int(f_idx),
                "timestamp_sec": round(float(t_sec), 3),
                "spoof_prob": round(float(f_prob), 4),
                "is_anomaly": is_anomaly,
            })

        # 5. Aggregate Score & Verdict
        overall_fake = float(np.mean([t["spoof_prob"] for t in timeline]))
        max_fake = float(np.max([t["spoof_prob"] for t in timeline]))
        # Sensitive aggregation
        final_fake = float(0.6 * overall_fake + 0.4 * max_fake)
        final_fake = float(np.clip(final_fake, 0.01, 0.99))
        final_real = float(round(1.0 - final_fake, 4))

        if final_fake >= 0.5:
            verdict = "FAKE"
            confidence = final_fake
        else:
            verdict = "REAL"
            confidence = final_real

        # 6. Extract Top-N Key Artifacts (Top 3 most suspicious frames with Grad-CAM)
        sorted_indices = np.argsort([t["spoof_prob"] for t in timeline])[::-1]
        top_k = min(3, len(sorted_indices))
        key_artifacts: List[Dict[str, Any]] = []

        for k in range(top_k):
            sel_idx = int(sorted_indices[k])
            sel_crop = frame_crops[sel_idx]
            sel_box = frame_bboxes[sel_idx]
            sel_time = timeline[sel_idx]

            # Generate Grad-CAM for this crop
            saliency_peak = [112, 112]
            if self.gradcam is not None and self.model is not None:
                try:
                    crop_tensor = self.normalizer.normalize(
                        self.tensor_converter.to_tensor([sel_crop])
                    ).unsqueeze(0).to(self.device)  # [1, 1, C, H, W]
                    heatmap = self.gradcam.generate_heatmap(crop_tensor, class_idx=1)
                    px, py = GradCAM.extract_peak_saliency(heatmap)
                    saliency_peak = [px, py]
                except Exception:
                    saliency_peak = [112, 112]

            key_artifacts.append({
                "frame_idx": sel_time["frame_idx"],
                "timestamp_sec": sel_time["timestamp_sec"],
                "bbox": list(sel_box) if sel_box else [0, 0, 224, 224],
                "spoof_prob": sel_time["spoof_prob"],
                "saliency_peak": saliency_peak,
            })

        return {
            "verdict": verdict,
            "confidence": round(float(confidence), 4),
            "raw_scores": {
                "real_prob": round(float(final_real), 4),
                "fake_prob": round(float(final_fake), 4),
            },
            "visual_cues": {
                "ela_discrepancy_score": round(avg_ela, 4),
                "fft_spectral_anomaly": round(avg_fft, 4),
                "boundary_inconsistency": round(avg_boundary, 4),
            },
            "timeline": timeline,
            "key_artifacts": key_artifacts,
        }

    def _fallback_result(self) -> Dict[str, Any]:
        """Default safe result on empty inputs."""
        return {
            "verdict": "REAL",
            "confidence": 0.5,
            "raw_scores": {"real_prob": 0.5, "fake_prob": 0.5},
            "visual_cues": {
                "ela_discrepancy_score": 0.5,
                "fft_spectral_anomaly": 0.5,
                "boundary_inconsistency": 0.5,
            },
            "timeline": [],
            "key_artifacts": [],
        }

    @property
    def is_ready(self) -> bool:
        """Whether detector is ready for inference."""
        return True


# Backward compatibility alias
VideoDetector = VideoDeepfakeDetector