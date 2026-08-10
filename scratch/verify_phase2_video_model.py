"""Phase 2 Video DeepFake Model Complete Verification Script."""

from __future__ import annotations

import gc
import hashlib
import os
import sys
import time
from pathlib import Path
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.video.configs.model_config import ModelConfig
from app.video.models.efficientnet.model import EfficientNetB4Model, ExecutionMode
from app.video.preprocessing.video_decoder import VideoDecoder
from app.video.preprocessing.frame_sampler import FrameSampler
from app.video.preprocessing.sequence_builder import SequenceBuilder
from app.video.preprocessing.video_tensor_converter import VideoTensorConverter
from app.video.preprocessing.video_normalizer import VideoNormalizer
from app.video.face_detection.face_detector import FaceDetector

FFPP_ORIGINAL_DIR = PROJECT_ROOT / "datasets" / "video" / "faceforensics" / "original"
AUDIO_CKPT = PROJECT_ROOT / "trained_models" / "audio" / "best_model.pt"

def assert_finite(tensor: torch.Tensor, name: str) -> None:
    assert torch.isfinite(tensor).all(), (
        f"[{name}] Contains NaN/Inf! NaN: {torch.isnan(tensor).sum().item()}, "
        f"Inf: {torch.isinf(tensor).sum().item()}"
    )

def run_verification():
    print("==================================================")
    print("  STARTING PHASE 2 VIDEO MODEL VERIFICATION REPORT")
    print("==================================================")

    # ----------------------------------------------------
    # 1. ARCHITECTURE & FEATURE DIMENSION AUDIT
    # ----------------------------------------------------
    print("\n--- 1. ARCHITECTURE AUDIT & TENSOR TRACE ---")
    config = ModelConfig(freeze_backbone=False)
    model = EfficientNetB4Model(config=config)
    model.eval()

    # Verify EfficientNet-B4 features dimension dynamically
    last_conv = model.backbone.net.features[-1][0]
    backbone_out_channels = last_conv.out_channels
    print(f"Backbone last conv channels: {backbone_out_channels}")
    assert backbone_out_channels == 1792, f"Expected 1792, got {backbone_out_channels}"
    assert model.backbone.feature_dim == 1792

    # Trace shapes
    x_test = torch.randn(2, 16, 3, 224, 224)
    print(f"Input shape:                    {list(x_test.shape)}")
    x_reshaped = x_test.view(2 * 16, 3, 224, 224)
    print(f"Reshaped shape (B*T, C, H, W):   {list(x_reshaped.shape)}")
    
    spatial_feats = model.backbone(x_reshaped)
    print(f"Backbone spatial feats shape:   {list(spatial_feats.shape)}")
    assert spatial_feats.shape == (32, 1792)

    seq_feats = spatial_feats.view(2, 16, 1792)
    print(f"Sequence feats shape (B, T, D): {list(seq_feats.shape)}")

    pos_feats = model.temporal_encoder.pos_encoder(seq_feats)
    print(f"Positional encoding shape:      {list(pos_feats.shape)}")

    trans_feats = pos_feats
    for block in model.temporal_encoder.blocks:
        trans_feats = block(trans_feats)
    print(f"Transformer encoder output:     {list(trans_feats.shape)}")

    clip_emb = model.temporal_encoder.pooling(trans_feats)
    print(f"Temporal pooling output:       {list(clip_emb.shape)}")
    assert clip_emb.shape == (2, 1792)

    attn_w = model.temporal_encoder.get_attention_weights(seq_feats)
    print(f"Temporal attention weights:     {list(attn_w.shape)}")
    assert attn_w.shape == (2, 16, 1)

    attn_sum = attn_w.sum(dim=1).squeeze(-1)
    print(f"Temporal attention sum over T:  {attn_sum.tolist()}")
    assert torch.allclose(attn_sum, torch.ones(2), atol=1e-5)

    logits = model.classifier(clip_emb)
    print(f"Classifier output (logits):     {list(logits.shape)}")
    assert logits.shape == (2, 2)

    print("Classifier structure:")
    print(model.classifier)

    # ----------------------------------------------------
    # 2. PARAMETER COUNTS & FREEZING
    # ----------------------------------------------------
    print("\n--- 2. PARAMETER COUNTS & FREEZING ---")
    model_unfrozen = EfficientNetB4Model(ModelConfig(freeze_backbone=False))
    total_params = model_unfrozen.get_num_parameters()
    trainable_unfrozen = model_unfrozen.get_trainable_parameters()

    model_frozen = EfficientNetB4Model(ModelConfig(freeze_backbone=True))
    trainable_frozen = model_frozen.get_trainable_parameters()
    frozen_params = total_params - trainable_frozen

    print(f"Total Parameters:      {total_params:,}")
    print(f"Unfrozen Trainable:    {trainable_unfrozen:,}")
    print(f"Frozen Trainable:      {trainable_frozen:,}")
    print(f"Frozen Parameters:     {frozen_params:,}")

    # Check backbone parameters requires_grad when frozen
    for p in model_frozen.backbone.parameters():
        assert not p.requires_grad
    # Check encoder & classifier parameters requires_grad when frozen
    for p in model_frozen.temporal_encoder.parameters():
        assert p.requires_grad
    for p in model_frozen.classifier.parameters():
        assert p.requires_grad
    print("Backbone freezing parameter check: PASSED")

    # ----------------------------------------------------
    # 3. SYNTHETIC FORWARD & BACKWARD TESTS
    # ----------------------------------------------------
    print("\n--- 3. SYNTHETIC FORWARD & BACKWARD TESTS ---")
    for b_size in [1, 2, 4]:
        x_synth = torch.randn(b_size, 16, 3, 224, 224)
        out = model_unfrozen(x_synth)
        assert out.shape == (b_size, 2)
        assert_finite(out, f"forward_B{b_size}")
        probs = F.softmax(out, dim=-1)
        assert_finite(probs, f"probs_B{b_size}")
        assert torch.allclose(probs.sum(dim=-1), torch.ones(b_size), atol=1e-5)
        print(f"Synthetic Forward B={b_size}: Output {list(out.shape)}, Dtype {out.dtype}, Prob sum {probs.sum(dim=-1).tolist()} [PASSED]")

    # Single-step backward test
    model_train = EfficientNetB4Model(ModelConfig(freeze_backbone=False))
    model_train.train()
    optimizer = torch.optim.AdamW(model_train.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    x_bwd = torch.randn(2, 16, 3, 224, 224)
    target_bwd = torch.tensor([0, 1], dtype=torch.long)

    optimizer.zero_grad()
    logits_bwd = model_train(x_bwd)
    loss_bwd = criterion(logits_bwd, target_bwd)
    assert_finite(loss_bwd, "synthetic_loss")
    loss_bwd.backward()

    for name, p in model_train.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Gradient missing for {name}"
            assert_finite(p.grad, f"grad_{name}")

    optimizer.step()
    print(f"Synthetic Single-Step Backward Test: Loss = {loss_bwd.item():.4f} [PASSED]")

    # ----------------------------------------------------
    # 4. AMP & CUDA BENCHMARKS (B=1, B=2, B=4)
    # ----------------------------------------------------
    print("\n--- 4. AMP & CUDA BENCHMARK ---")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU Device: {gpu_name}")

        benchmark_results = {}
        for b_size in [1, 2, 4]:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            m = EfficientNetB4Model(ModelConfig(freeze_backbone=False)).to(device)
            m.train()
            opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
            scaler = torch.amp.GradScaler("cuda")
            crit = nn.CrossEntropyLoss()

            x_gpu = torch.randn(b_size, 16, 3, 224, 224, device=device)
            y_gpu = torch.randint(0, 2, (b_size,), device=device, dtype=torch.long)

            # Warmup
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                out_gpu = m(x_gpu)
                l_gpu = crit(out_gpu, y_gpu)
            scaler.scale(l_gpu).backward()
            scaler.step(opt)
            scaler.update()

            torch.cuda.reset_peak_memory_stats()

            # Forward timing
            m.eval()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    out_fwd = m(x_gpu)
            torch.cuda.synchronize()
            t_fwd = (time.perf_counter() - t0) * 1000.0

            # Backward timing
            m.train()
            opt.zero_grad()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.amp.autocast("cuda"):
                out_bwd = m(x_gpu)
                l_bwd = crit(out_bwd, y_gpu)
            scaler.scale(l_bwd).backward()
            torch.cuda.synchronize()
            t_bwd = (time.perf_counter() - t0) * 1000.0

            input_bytes = x_gpu.element_size() * x_gpu.nelement()
            input_mb = input_bytes / (1024 ** 2)
            model_mb = sum(p.element_size() * p.nelement() for p in m.parameters()) / (1024 ** 2)
            peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_res_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

            benchmark_results[b_size] = {
                "input_mb": input_mb,
                "model_mb": model_mb,
                "peak_alloc_mb": peak_alloc_mb,
                "peak_res_mb": peak_res_mb,
                "fwd_ms": t_fwd,
                "bwd_ms": t_bwd,
            }

            print(f"B={b_size}: Peak Alloc = {peak_alloc_mb:.2f} MB, Peak Res = {peak_res_mb:.2f} MB, Fwd = {t_fwd:.2f} ms, Bwd = {t_bwd:.2f} ms")

            del m, opt, scaler, x_gpu, y_gpu, out_fwd, out_bwd, l_bwd
            gc.collect()
            torch.cuda.empty_cache()

    # ----------------------------------------------------
    # 5. REAL VIDEO INTEGRATION TEST
    # ----------------------------------------------------
    print("\n--- 5. REAL VIDEO INTEGRATION TEST ---")
    mp4_files = sorted(FFPP_ORIGINAL_DIR.glob("*.mp4"))
    if len(mp4_files) > 0:
        vpath = mp4_files[0]
        print(f"Testing on real video: {vpath.name}")
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
        tensor_16 = normalizer.normalize(converter.to_tensor(seq_arr))
        input_batch = tensor_16.unsqueeze(0)
        assert input_batch.shape == (1, 16, 3, 224, 224)
        assert_finite(input_batch, "real_video_tensor")

        eval_model = EfficientNetB4Model(ModelConfig(pretrained=False))
        eval_model.eval()
        with torch.no_grad():
            real_logits = eval_model(input_batch)
            real_probs = F.softmax(real_logits, dim=-1)
            real_attn = eval_model.get_attention_weights(input_batch)

        assert real_logits.shape == (1, 2)
        assert real_probs.shape == (1, 2)
        assert real_attn.shape == (1, 16, 1)
        assert_finite(real_logits, "real_logits")
        assert_finite(real_probs, "real_probs")
        assert_finite(real_attn, "real_attn")
        print(f"Real video inference: Logits = {real_logits.numpy().round(4)}, Probs = {real_probs.numpy().round(4)} [PASSED]")
    else:
        print("No real video file found at FFPP_ORIGINAL_DIR.")

    # ----------------------------------------------------
    # 6. CHECKPOINT SAFETY CHECK
    # ----------------------------------------------------
    print("\n--- 6. CHECKPOINT SAFETY CHECK ---")
    if AUDIO_CKPT.exists():
        stat = AUDIO_CKPT.stat()
        print(f"Audio Checkpoint Exists: {AUDIO_CKPT.name}, Size: {stat.st_size} bytes, Modified: {stat.st_mtime}")
    else:
        print(f"Audio Checkpoint not found at {AUDIO_CKPT}")

    print("\n==================================================")
    print("  PHASE 2 VIDEO MODEL VERIFICATION SCRIPT COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    run_verification()
