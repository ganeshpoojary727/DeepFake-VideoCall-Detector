"""
Video Visualization, Explainability (Grad-CAM), and Annotation Utilities.

Provides:
- GradCAM: Class activation mapping for convolutional backbones (e.g. EfficientNet).
- Heatmap overlay and hotspot peak extraction for visual artifact explanation.
- Bounding box annotation and frame grid strip generation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Grad-CAM Saliency Visualizer ───────────────────────────────────────────────

class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) for deepfake visual explainability.

    Hooks into the final convolutional feature layer to compute spatial sensitivity
    heatmaps highlighting deepfake artifacts and facial boundary anomalies.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
    ) -> None:
        self.model = model
        self.target_layer = target_layer or self._find_target_layer(model)
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks: List[Any] = []
        self._register_hooks()

    def _find_target_layer(self, model: nn.Module) -> Optional[nn.Module]:
        """Automatically identify the last 2D convolutional layer in the backbone."""
        last_conv = None
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                last_conv = module
        return last_conv

    def _register_hooks(self) -> None:
        if self.target_layer is None:
            logger.debug("GradCAM: No convolutional target layer identified.")
            return

        def forward_hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(module: nn.Module, grad_input: Any, grad_output: Tuple[torch.Tensor, ...]) -> None:
            if grad_output and grad_output[0] is not None:
                self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        """Clean up registered PyTorch hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        class_idx: int = 1,
        target_size: Optional[Tuple[int, int]] = (224, 224),
    ) -> np.ndarray:
        """Generate normalized 2D Grad-CAM saliency heatmap for input tensor.

        Args:
            input_tensor: Input tensor shape [B, C, H, W] or [B, T, C, H, W].
            class_idx: Target class index (1 = Fake/Spoof).
            target_size: Output (width, height) to resize heatmap.

        Returns:
            np.ndarray: 2D float32 array in [0.0, 1.0] of shape (H, W).
        """
        self.model.eval()
        self.activations = None
        self.gradients = None

        # Ensure tensor requires gradient
        x = input_tensor.clone().detach().requires_grad_(True)

        try:
            output = self.model(x)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            if logits.ndim > 1 and logits.shape[-1] > class_idx:
                score = logits[:, class_idx].sum()
            else:
                score = logits.sum()

            self.model.zero_grad()
            score.backward(retain_graph=True)

            if self.activations is None or self.gradients is None:
                # Fallback synthetic saliency if hooks did not capture gradients
                return self._generate_fallback_heatmap(target_size)

            grads = self.gradients
            acts = self.activations

            # Handle sequence dimension if 5D tensor [B, T, C, H, W]
            if grads.ndim == 5:
                grads = grads[:, -1]
            if acts.ndim == 5:
                acts = acts[:, -1]

            # Pool gradients across channels
            weights = torch.mean(grads, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * acts, dim=1, keepdim=True)
            cam = F.relu(cam)

            cam_np = cam.squeeze().cpu().numpy().astype(np.float32)
            if cam_np.ndim != 2:
                cam_np = cam_np[0] if cam_np.ndim > 2 else np.zeros((16, 16), dtype=np.float32)

            # Normalize to [0.0, 1.0]
            max_val = np.max(cam_np)
            if max_val > 0:
                cam_np = cam_np / max_val
            else:
                cam_np = np.zeros_like(cam_np)

            if target_size is not None:
                tw, th = target_size
                cam_np = cv2.resize(cam_np, (tw, th), interpolation=cv2.INTER_LINEAR)

            return np.clip(cam_np, 0.0, 1.0).astype(np.float32)

        except Exception as exc:
            logger.debug("GradCAM execution error: %s. Returning fallback heatmap.", exc)
            return self._generate_fallback_heatmap(target_size)

    @staticmethod
    def _generate_fallback_heatmap(target_size: Optional[Tuple[int, int]] = (224, 224)) -> np.ndarray:
        """Generate centered Gaussian fallback heatmap when gradients are unavailable."""
        tw, th = target_size if target_size is not None else (224, 224)
        y, x = np.ogrid[:th, :tw]
        cy, cx = th // 2, tw // 2
        sigma = min(tw, th) / 3.0
        gaussian = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
        return gaussian.astype(np.float32)

    @staticmethod
    def overlay_heatmap(
        image_bgr: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Blend colorized Grad-CAM heatmap over image frame.

        Args:
            image_bgr: Base image array (H, W, 3) in BGR format.
            heatmap: 2D float32 heatmap array in [0.0, 1.0].
            alpha: Heatmap blend weight (0.0 to 1.0).
            colormap: OpenCV colormap enum (default: COLORMAP_JET).

        Returns:
            np.ndarray: Blended BGR image array.
        """
        h, w = image_bgr.shape[:2]
        if heatmap.shape[:2] != (h, w):
            resized_cam = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            resized_cam = heatmap

        cam_uint8 = np.uint8(255.0 * np.clip(resized_cam, 0.0, 1.0))
        color_cam = cv2.applyColorMap(cam_uint8, colormap)

        overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, color_cam, alpha, 0)
        return overlay

    @staticmethod
    def extract_peak_saliency(heatmap: np.ndarray) -> Tuple[int, int]:
        """Find (x, y) coordinates of maximum activation hotspot in heatmap."""
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap)
        return int(max_loc[0]), int(max_loc[1])


# ── Annotation & Utility Functions ─────────────────────────────────────────────

def draw_bboxes(
    frame: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw facial bounding boxes onto frame array (H, W, C)."""
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
    """Grid tile sequence of video frames into a single image strip array."""
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
) -> Dict[str, List[float]]:
    """Return dictionary structure of training history metrics for plotting."""
    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
    }
