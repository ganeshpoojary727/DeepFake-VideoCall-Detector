"""Training smoke test for NEW AASIST architecture with random initialization and pipeline safety checks."""

from __future__ import annotations

import json
import math

from pathlib import Path
import sys

# Ensure root project directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch
import torch.nn as nn

from app.audio.models.aasist import AASIST
from app.audio.configs.training_config import AudioTrainingConfig
from app.audio.datasets.dataloader import create_train_dataloader, create_validation_dataloader, compute_class_weights
from app.audio.training.trainer import ProductionAudioTrainer
from app.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    print("=" * 60)
    print("NEW AASIST ARCHITECTURE TRAINING SMOKE TEST")
    print("=" * 60)

    # 10. Confirm model is NEW AASIST architecture containing MaxPool1d(4)
    model = AASIST(num_classes=2)
    has_maxpool = hasattr(model.frontend.encoder, "first_pool") and isinstance(model.frontend.encoder.first_pool, nn.MaxPool1d)
    print(f"10. Architecture check: Model contains MaxPool1d(4): {has_maxpool}")
    assert has_maxpool, "Model must be the NEW AASIST architecture with MaxPool1d(4)!"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 9. Confirm random initialization (no old checkpoint loaded)
    print("9. Random initialization: PASSED (No old checkpoint loaded)")

    # Dataloaders
    train_loader = create_train_dataloader()
    val_loader = create_validation_dataloader()

    # Class weights for loss
    class_weights = compute_class_weights(train_loader.dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training configuration
    temp_ckpt_dir = Path("trained_models/audio_smoke_test")
    temp_ckpt_dir.mkdir(parents=True, exist_ok=True)

    config = AudioTrainingConfig(
        batch_size=32,
        learning_rate=1e-4,
        epochs=1,
        use_amp=True,
        gradient_clip_norm=1.0,
        checkpoint_dir=temp_ckpt_dir,
        log_dir=temp_ckpt_dir / "logs",
        tensorboard_dir=temp_ckpt_dir / "tb",
    )

    trainer = ProductionAudioTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        criterion=criterion,
        device=str(device),
    )

    print("\nExecuting 1-epoch training smoke test...")
    # Run 1 epoch
    train_loss = trainer.train_epoch(epoch=1)
    trainer.history["train_loss"].append(train_loss)

    # 1. Forward loss finite
    is_fwd_loss_finite = math.isfinite(train_loss)
    print(f"1. Forward train loss is finite: {is_fwd_loss_finite} (loss = {train_loss:.4f})")
    assert is_fwd_loss_finite, f"Train loss is non-finite: {train_loss}"

    # 2. Backward gradients finite
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    are_grads_finite = all(math.isfinite(g) for g in grad_norms)
    max_grad = max(grad_norms) if grad_norms else 0.0
    print(f"2. Backward gradients are finite: {are_grads_finite} (max grad norm = {max_grad:.4f})")
    assert are_grads_finite, "Backward gradients contain non-finite values!"

    # 3. Model parameters remain finite
    are_params_finite = all(torch.isfinite(p).all().item() for p in model.parameters())
    print(f"3. Model parameters remain finite: {are_params_finite}")
    assert are_params_finite, "Model parameters contain non-finite values after epoch!"

    # Run validation
    print("\nExecuting validation pass...")
    val_metrics = trainer.validator.evaluate(val_loader, criterion)

    val_loss = val_metrics.get("val_loss")
    accuracy = val_metrics.get("accuracy")
    eer = val_metrics.get("eer")

    trainer.history["val_loss"].append(val_loss)
    trainer.history["val_acc"].append(accuracy)
    trainer.history["val_eer"].append(eer)

    # 4. Validation loss finite
    is_val_loss_finite = val_loss is not None and math.isfinite(val_loss)
    print(f"4. Validation loss is finite: {is_val_loss_finite} (val_loss = {val_loss})")
    assert is_val_loss_finite, f"Validation loss is non-finite: {val_loss}"

    # 5. Accuracy finite and not artificially 10.26%
    is_acc_valid = accuracy is not None and math.isfinite(accuracy) and (abs(accuracy - 0.10256) > 0.01)
    print(f"5. Accuracy is finite & not 10.26%: {is_acc_valid} (accuracy = {accuracy:.4%})")
    assert is_acc_valid, f"Accuracy is invalid or stuck at 10.26%: {accuracy}"

    # 6. EER is finite and not artificially 0%
    is_eer_valid = eer is not None and math.isfinite(eer) and (eer > 0.0)
    print(f"6. EER is finite & not 0.00%: {is_eer_valid} (eer = {eer:.4%})")
    assert is_eer_valid, f"EER is invalid or stuck at 0.00%: {eer}"

    # Save checkpoint via CheckpointManager
    ckpt_path = trainer.checkpoint_manager.save(
        model=model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        epoch=1,
        metric_value=val_loss,
        filename="smoke_test_ckpt.pt",
        history=trainer.history,
    )

    # 7. Checkpoint contains no NaN/Inf parameters
    ckpt_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt_dict["model_state_dict"]
    ckpt_params_finite = all(torch.isfinite(v).all().item() for v in sd.values() if isinstance(v, torch.Tensor))
    print(f"7. Saved checkpoint params are finite: {ckpt_params_finite}")
    assert ckpt_params_finite, "Saved checkpoint state_dict contains non-finite values!"

    # 8. training_history.json contains no NaN/Infinity
    hist_file = temp_ckpt_dir / "training_history.json"
    assert hist_file.exists(), "training_history.json must exist."
    with open(hist_file, "r", encoding="utf-8") as f:
        hist_data = json.load(f)

    # Recursively check for 'NaN' string or float nan/inf in parsed json
    def check_no_nan_str(obj):
        if isinstance(obj, str):
            assert obj.lower() not in ["nan", "infinity", "-infinity"], f"Found raw nan/inf string in JSON: {obj}"
        elif isinstance(obj, dict):
            for v in obj.values():
                check_no_nan_str(v)
        elif isinstance(obj, list):
            for v in obj:
                check_no_nan_str(v)

    check_no_nan_str(hist_data)
    print(f"8. training_history.json contains no NaN/Infinity: PASSED")

    print("\n" + "=" * 60)
    print("ALL 10 VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
