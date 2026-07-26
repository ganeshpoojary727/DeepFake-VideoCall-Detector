import torch

from app.ai.models.cnn_model import DeepFakeCNN
from app.ai.datasets.dataloader import create_test_dataloader
from app.ai.evaluation.evaluator import Evaluator

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using Device: {device}")

print("Loading model...")

model = DeepFakeCNN().to(device)

print("Loading weights...")

model.load_state_dict(
    torch.load(
        "trained_models/best_model.pth",
        map_location=device
    )
)

print("Creating test dataloader...")

test_loader = create_test_dataloader()

print("Creating evaluator...")

evaluator = Evaluator(
    model=model,
    test_loader=test_loader,
    device=device
)

print("Starting evaluation...")

accuracy, precision, recall, f1, matrix, report = evaluator.evaluate()

print("Evaluation finished!")