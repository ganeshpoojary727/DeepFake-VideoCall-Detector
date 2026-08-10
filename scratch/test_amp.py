"""Test AMP GradScaler with clip_grad_norm_ and scale adjustment."""

import sys
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.video.models.efficientnet.model import EfficientNetB4Model
from app.video.configs.model_config import ModelConfig

def main():
    device = torch.device("cuda:0")
    model = EfficientNetB4Model(ModelConfig(freeze_backbone=True)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(4, 16, 3, 224, 224, device=device)
    y = torch.tensor([0, 1, 0, 1], device=device)

    print("Initial Scaler Scale:", scaler.get_scale())

    for step in range(5):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        new_scale = scaler.get_scale()

        print(f"Step {step+1}: Loss = {loss.item():.4f}, GradNorm = {grad_norm.item():.4f}, Old Scale = {old_scale}, New Scale = {new_scale}")

if __name__ == "__main__":
    main()
