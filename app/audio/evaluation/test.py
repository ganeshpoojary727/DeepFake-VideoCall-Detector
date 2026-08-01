"""
Evaluation script — run model evaluation on the test set.

Usage::

    python -m app.audio.evaluation.test

Improvements over v1
─────────────────────
• Wrapped in ``main()`` + ``if __name__`` guard
• Uses ``Settings`` for model path (no hardcoded string)
• **Prints evaluation results** (v1 computed but never displayed them)
• Shows EER alongside standard metrics
"""

from __future__ import annotations

import torch

from app.audio.datasets.dataloader import create_test_dataloader
from app.audio.models.legacy_cnn_model import DeepFakeCNN
from app.audio.evaluation.evaluator import Evaluator
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Run evaluation on the test set and display results."""

    device = settings.DEVICE
    logger.info("Using device: %s", device)

    # ── Load model ────────────────────────────
    model_path = settings.MODEL_SAVE_PATH
    if not model_path.exists():
        logger.error("Model not found at %s — train first.", model_path)
        return

    logger.info("Loading model from %s", model_path)
    model = DeepFakeCNN(num_classes=settings.model.num_classes).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )

    # ── DataLoader ────────────────────────────
    logger.info("Creating test dataloader...")
    test_loader = create_test_dataloader()

    # ── Evaluate ──────────────────────────────
    logger.info("Starting evaluation...")
    evaluator = Evaluator(model=model, test_loader=test_loader, device=device)
    result = evaluator.evaluate()

    # ── Display results ───────────────────────
    print("\n" + "=" * 50)
    print("         EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy:  {result.accuracy:.4f}  ({result.accuracy * 100:.2f}%)")
    print(f"  Precision: {result.precision:.4f}")
    print(f"  Recall:    {result.recall:.4f}")
    print(f"  F1 Score:  {result.f1:.4f}")
    if result.eer is not None:
        print(f"  EER:       {result.eer:.4f}  ({result.eer * 100:.2f}%)")
    print("-" * 50)
    print("  Confusion Matrix:")
    print(f"  {result.confusion_matrix}")
    print("-" * 50)
    print(result.classification_report)
    print("=" * 50)


if __name__ == "__main__":
    main()