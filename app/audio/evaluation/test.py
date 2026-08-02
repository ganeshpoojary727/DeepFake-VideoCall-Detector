"""Evaluation script — run AASIST model evaluation on test set."""

from __future__ import annotations

import torch

from app.audio.datasets.dataloader import create_test_dataloader
from app.audio.evaluation.evaluator import Evaluator
from app.audio.models.aasist import AASIST
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Run evaluation on the test set and display results."""
    device = settings.DEVICE
    logger.info("Using device: %s", device)

    model_path = settings.MODEL_SAVE_PATH
    if not model_path.exists():
        logger.error("Model not found at %s — train first.", model_path)
        return

    logger.info("Loading AASIST model from %s", model_path)
    model = AASIST(num_classes=settings.model.num_classes).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )

    logger.info("Creating test dataloader...")
    test_loader = create_test_dataloader()

    logger.info("Starting evaluation...")
    evaluator = Evaluator(model=model, test_loader=test_loader, device=device)
    result = evaluator.evaluate()

    print("\n" + "=" * 50)
    print("         EVALUATION RESULTS (AASIST)")
    print("=" * 50)
    print(f"  Accuracy:  {result.accuracy:.4f}  ({result.accuracy * 100:.2f}%)")


if __name__ == "__main__":
    main()