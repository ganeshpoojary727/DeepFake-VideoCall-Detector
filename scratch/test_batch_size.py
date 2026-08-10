"""Test VRAM and speed for Batch Size 4 vs 8."""

import sys
import time
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.video.models.efficientnet.model import EfficientNetB4Model
from app.video.configs.model_config import ModelConfig

def main():
    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    for bs in [4, 8]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        model = EfficientNetB4Model(ModelConfig(freeze_backbone=True)).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        criterion = torch.nn.CrossEntropyLoss()

        x = torch.randn(bs, 16, 3, 224, 224, device=device)
        y = torch.randint(0, 2, (bs,), device=device, dtype=torch.long)

        # Forward + Backward
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"Batch Size {bs}: Time = {(t1 - t0)*1000:.2f} ms, Peak VRAM = {vram_mb:.2f} MB")

if __name__ == "__main__":
    main()
