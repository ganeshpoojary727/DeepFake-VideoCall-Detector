"""Phase 2 Video DeepFake Model Architecture & Smoke Test.

This test suite verifies the complete Video DeepFake Model (EfficientNetB4Model):
  1. Model instantiation with default and custom configs
  2. Forward pass with synthetic video tensor [2, 16, 3, 224, 224]
  3. Logits shape [2, 2] and float32 dtype
  4. Finite logits verification (no NaN / Inf)
  5. Softmax probability computation and sum-to-1 validation
  6. Temporal attention weights extraction, dimension check [B, T, 1], and sum-to-1 over T
  7. Backbone freezing strategy (freeze_backbone=True vs False trainable parameter counts)
  8. Loss calculation with CrossEntropyLoss
  9. Backward pass and finite gradient verification
 10. Optimizer step execution (1 complete training step)
 11. CUDA & AMP mixed precision execution (if GPU available)
 12. GPU memory profiling & latency benchmark (params, VRAM MB, forward/backward ms)
 13. Phase 1 -> Phase 2 integration test (real .mp4 -> Phase 1 pipeline -> model logits -> softmax)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.video.configs.model_config import ModelConfig
from app.video.models.efficientnet.model import EfficientNetB4Model, ExecutionMode

# -- Project Root & Dataset Paths --------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FFPP_ORIGINAL_DIR = PROJECT_ROOT / "datasets" / "video" / "faceforensics" / "original"

# -- Helpers -----------------------------------------------------------------

def _assert_finite(tensor: torch.Tensor, tag: str) -> None:
    """Assert that a tensor has NO NaN or Inf values."""
    assert torch.isfinite(tensor).all(), (
        f"[{tag}] Tensor contains NaN or Inf! "
        f"NaN count: {torch.isnan(tensor).sum().item()}, "
        f"Inf count: {torch.isinf(tensor).sum().item()}, "
        f"shape: {tensor.shape}"
    )


# ------------------------------------------------------------------------------
# 1. Model Instantiation & Configuration Tests
# ------------------------------------------------------------------------------

class TestModelInstantiation:
    """Verify model instantiation, parameter counts, and freezing modes."""

    def test_default_instantiation(self) -> None:
        model = EfficientNetB4Model()
        assert model is not None
        assert isinstance(model, nn.Module)
        assert model.config.num_classes == 2
        assert model.config.in_channels == 3
        print(f"  [INIT] Default model params: {model.get_num_parameters():,}")

    def test_freeze_backbone_true(self) -> None:
        cfg = ModelConfig(freeze_backbone=True)
        model = EfficientNetB4Model(config=cfg)
        total = model.get_num_parameters()
        trainable = model.get_trainable_parameters()
        assert trainable < total, (
            f"Trainable params ({trainable}) should be less than total ({total}) when frozen"
        )
        print(f"  [FREEZE=True] Total: {total:,}, Trainable: {trainable:,}")

    def test_freeze_backbone_false(self) -> None:
        cfg = ModelConfig(freeze_backbone=False)
        model = EfficientNetB4Model(config=cfg)
        total = model.get_num_parameters()
        trainable = model.get_trainable_parameters()
        assert trainable == total, (
            f"Trainable params ({trainable}) should equal total ({total}) when unfrozen"
        )
        print(f"  [FREEZE=False] Total: {total:,}, Trainable: {trainable:,}")


# ------------------------------------------------------------------------------
# 2. Forward Pass & Probability Validation (Synthetic Tensor)
# ------------------------------------------------------------------------------

class TestForwardPassSynthetic:
    """Run forward pass on synthetic tensor [2, 16, 3, 224, 224]."""

    @pytest.fixture
    def synthetic_input(self) -> torch.Tensor:
        """Create synthetic video input tensor [2, 16, 3, 224, 224]."""
        torch.manual_seed(42)
        return torch.randn(2, 16, 3, 224, 224, dtype=torch.float32)

    @pytest.fixture
    def model(self) -> EfficientNetB4Model:
        model = EfficientNetB4Model(ModelConfig(pretrained=False, freeze_backbone=False))
        model.eval()
        return model

    def test_forward_output_shape(self, model, synthetic_input) -> None:
        with torch.no_grad():
            logits = model(synthetic_input)
        assert logits.shape == (2, 2), f"Expected logits shape (2, 2), got {logits.shape}"
        print(f"  [FORWARD] Logits shape: {logits.shape} [OK]")

    def test_forward_output_dtype(self, model, synthetic_input) -> None:
        with torch.no_grad():
            logits = model(synthetic_input)
        assert logits.dtype == torch.float32, f"Expected float32, got {logits.dtype}"

    def test_forward_logits_are_finite(self, model, synthetic_input) -> None:
        with torch.no_grad():
            logits = model(synthetic_input)
        _assert_finite(logits, "logits")
        print(f"  [FORWARD] Logits finite check [OK]")

    def test_softmax_probabilities_valid(self, model, synthetic_input) -> None:
        with torch.no_grad():
            logits = model(synthetic_input)
            probs = F.softmax(logits, dim=-1)

        _assert_finite(probs, "softmax_probs")
        assert probs.min() >= 0.0, f"Min prob {probs.min():.4f} < 0"
        assert probs.max() <= 1.0, f"Max prob {probs.max():.4f} > 1"

        # Check sum over class dimension equals 1.0 for each sample
        prob_sums = probs.sum(dim=-1)
        expected_ones = torch.ones(2, dtype=torch.float32)
        assert torch.allclose(prob_sums, expected_ones, atol=1e-5), (
            f"Probabilities sum to {prob_sums.tolist()}, expected [1.0, 1.0]"
        )
        print(f"  [SOFTMAX] Probabilities sum to 1.0: {prob_sums.tolist()} [OK]")


# ------------------------------------------------------------------------------
# 3. Temporal Attention Weights Validation
# ------------------------------------------------------------------------------

class TestTemporalAttentionWeights:
    """Verify extraction and properties of temporal attention weights."""

    def test_attention_weights_extraction(self) -> None:
        model = EfficientNetB4Model(ModelConfig(pretrained=False))
        model.eval()

        x = torch.randn(2, 16, 3, 224, 224)
        with torch.no_grad():
            attn_weights = model.get_attention_weights(x)

        # Expected shape (2, 16, 1) or (2, 16)
        assert attn_weights.ndim in (2, 3), f"Attention weights dim {attn_weights.ndim} invalid"
        assert attn_weights.size(0) == 2, f"Batch dim {attn_weights.size(0)} != 2"
        assert attn_weights.size(1) == 16, f"Temporal dim {attn_weights.size(1)} != 16"

        _assert_finite(attn_weights, "attention_weights")

        # Sum over temporal dimension should equal 1.0
        if attn_weights.ndim == 3:
            attn_sums = attn_weights.sum(dim=1).squeeze(-1)  # (B,)
        else:
            attn_sums = attn_weights.sum(dim=1)  # (B,)

        expected = torch.ones(2, dtype=torch.float32)
        assert torch.allclose(attn_sums, expected, atol=1e-5), (
            f"Attention weights sum to {attn_sums.tolist()} over temporal dim, expected [1.0, 1.0]"
        )
        print(f"  [ATTENTION] Weights shape: {attn_weights.shape}, sum over T: {attn_sums.tolist()} [OK]")


# ------------------------------------------------------------------------------
# 4. Backward Pass & Training Step Verification
# ------------------------------------------------------------------------------

class TestBackwardPassAndTrainingStep:
    """Verify loss calculation, backward pass, finite gradients, and optimizer step."""

    def test_full_training_step(self) -> None:
        model = EfficientNetB4Model(ModelConfig(pretrained=False, freeze_backbone=False))
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(2, 16, 3, 224, 224, requires_grad=False)
        target = torch.tensor([0, 1], dtype=torch.long)

        # Forward
        optimizer.zero_grad()
        logits = model(x)
        assert logits.shape == (2, 2)

        # Loss
        loss = criterion(logits, target)
        _assert_finite(loss, "loss")
        assert loss.item() > 0.0

        # Backward
        loss.backward()

        # Check gradients exist and are finite for all trainable parameters
        trainable_grads = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has None grad"
                _assert_finite(param.grad, f"grad_{name}")
                trainable_grads += 1

        assert trainable_grads > 0, "No trainable gradients found"

        # Optimizer step
        optimizer.step()
        print(f"  [TRAIN STEP] Loss: {loss.item():.4f}, trainable grads checked: {trainable_grads} [OK]")


# ------------------------------------------------------------------------------
# 5. CUDA & AMP Mixed Precision Execution (if GPU available)
# ------------------------------------------------------------------------------

class TestCUDAAndAMPExecution:
    """Verify model transfer and mixed precision execution on CUDA GPU."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU available")
    def test_cuda_amp_forward_backward(self) -> None:
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        device = torch.device("cuda:0")
        model = EfficientNetB4Model(ModelConfig(pretrained=False, freeze_backbone=False)).to(device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(2, 16, 3, 224, 224, device=device)
        target = torch.tensor([0, 1], device=device, dtype=torch.long)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(x)
            loss = criterion(logits, target)

        assert logits.device.type == "cuda"
        _assert_finite(logits, "cuda_logits")
        _assert_finite(loss, "cuda_loss")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        del model, x, target, logits, loss
        gc.collect()
        torch.cuda.empty_cache()

        print(f"  [CUDA AMP] Loss: execution [OK]")


# ------------------------------------------------------------------------------
# 6. RTX 4050 Memory & Latency Profiling
# ------------------------------------------------------------------------------

class TestMemoryAndLatencyProfiling:
    """Profile parameters, input memory, peak VRAM, and forward/backward latencies."""

    def test_profile_model(self) -> None:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = EfficientNetB4Model(ModelConfig(pretrained=False, freeze_backbone=False))
        total_params = model.get_num_parameters()
        trainable_params = model.get_trainable_parameters()

        # Input tensor memory calculation
        # [2, 16, 3, 224, 224] float32 = 2 * 16 * 3 * 224 * 224 * 4 bytes
        input_numel = 2 * 16 * 3 * 224 * 224
        input_mem_mb = (input_numel * 4) / (1024 ** 2)

        # Benchmark latencies on current device
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)
        model = model.to(device)
        x = torch.randn(2, 16, 3, 224, 224, device=device)
        target = torch.tensor([0, 1], device=device, dtype=torch.long)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Warmup
        model.train()
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        # Forward latency timing
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            logits = model(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_fwd = (time.perf_counter() - t0) * 1000.0  # ms

        # Backward latency timing
        model.train()
        optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        logits = model(x)
        loss = criterion(logits, target)
        loss.backward()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_bwd = (time.perf_counter() - t0) * 1000.0  # ms

        # Memory usage
        peak_gpu_mem_mb = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
            if torch.cuda.is_available()
            else 0.0
        )

        del model, x, target, logits, loss
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            f"\n  +----------------------------------------------+\n"
            f"  |  PHASE 2 MODEL MEMORY & LATENCY PROFILE      |\n"
            f"  +----------------------------------------------+\n"
            f"  |  Device:           {device_str:<25s} |\n"
            f"  |  Total Params:     {total_params:<25,d} |\n"
            f"  |  Trainable Params: {trainable_params:<25,d} |\n"
            f"  |  Input Tensor Mem: {input_mem_mb:<22.2f} MB |\n"
            f"  |  Peak GPU VRAM:    {peak_gpu_mem_mb:<22.2f} MB |\n"
            f"  |  Forward Latency:  {t_fwd:<22.2f} ms |\n"
            f"  |  Backward Latency: {t_bwd:<22.2f} ms |\n"
            f"  +----------------------------------------------+"
        )

        assert total_params > 0
        assert trainable_params > 0



# ------------------------------------------------------------------------------
# 7. Phase 1 -> Phase 2 Integration Test (Real Video -> Model Logits)
# ------------------------------------------------------------------------------

class TestPhase1ToPhase2Integration:
    """Integration test connecting Phase 1 preprocessing with Phase 2 model."""

    def test_real_video_to_model_logits(self) -> None:
        """Full pipeline: real mp4 -> Phase 1 decode/crop/normalize -> Phase 2 model -> logits."""
        import cv2
        from app.video.preprocessing.video_decoder import VideoDecoder
        from app.video.preprocessing.frame_sampler import FrameSampler
        from app.video.preprocessing.sequence_builder import SequenceBuilder
        from app.video.preprocessing.video_tensor_converter import VideoTensorConverter
        from app.video.preprocessing.video_normalizer import VideoNormalizer
        from app.video.face_detection.face_detector import FaceDetector

        mp4_files = sorted(FFPP_ORIGINAL_DIR.glob("*.mp4"))
        assert len(mp4_files) > 0, "No FF++ original mp4 files found for integration test"
        vpath = mp4_files[0]

        # Phase 1: Preprocessing
        decoder = VideoDecoder()
        sampler = FrameSampler(num_frames=16, strategy="uniform")
        detector = FaceDetector(conf_threshold=0.5)
        builder = SequenceBuilder(sequence_length=16)
        converter = VideoTensorConverter()
        normalizer = VideoNormalizer()

        all_frames = decoder.decode(str(vpath))
        sampled = sampler.sample(all_frames)

        crops = []
        for f in sampled:
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            box = detector.detect_largest(bgr)
            if box:
                crops.append(detector.crop_face(bgr, box, target_size=(224, 224)))
            else:
                h, w = f.shape[:2]
                s = min(h, w)
                crops.append(cv2.resize(f[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2], (224, 224)))

        seq_arr = builder.build(crops)
        tensor_16 = normalizer.normalize(converter.to_tensor(seq_arr))  # (16, 3, 224, 224)
        input_batch = tensor_16.unsqueeze(0)  # (1, 16, 3, 224, 224)

        _assert_finite(input_batch, "integration_input")

        # Phase 2: Model Execution
        model = EfficientNetB4Model(ModelConfig(pretrained=False))
        model.eval()

        with torch.no_grad():
            logits = model(input_batch)  # (1, 2)
            probs = F.softmax(logits, dim=-1)  # (1, 2)
            attn_weights = model.get_attention_weights(input_batch)

        assert logits.shape == (1, 2), f"Expected logits (1, 2), got {logits.shape}"
        assert probs.shape == (1, 2), f"Expected probs (1, 2), got {probs.shape}"

        _assert_finite(logits, "integration_logits")
        _assert_finite(probs, "integration_probs")
        _assert_finite(attn_weights, "integration_attn")

        prob_real = probs[0, 0].item()
        prob_fake = probs[0, 1].item()

        print(
            f"\n  +----------------------------------------------+\n"
            f"  |  PHASE 1 -> PHASE 2 INTEGRATION TEST         |\n"
            f"  +----------------------------------------------+\n"
            f"  |  Input Video: {vpath.name:<30s} |\n"
            f"  |  Batch Shape: {str(input_batch.shape):<30s} |\n"
            f"  |  Logits:      {str(logits.detach().numpy().round(4)):<30s} |\n"
            f"  |  Prob (Real): {prob_real:<30.4f} |\n"
            f"  |  Prob (Fake): {prob_fake:<30.4f} |\n"
            f"  |  Integration: SUCCESS [OK]                  |\n"
            f"  +----------------------------------------------+"
        )
