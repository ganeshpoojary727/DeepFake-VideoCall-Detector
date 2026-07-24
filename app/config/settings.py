from pathlib import Path
import torch


class Settings:

    def __init__(self):

        self.PROJECT_ROOT = Path(__file__).resolve().parents[2]

        self.DATASET_DIR = (
    self.PROJECT_ROOT
    / "app"
    / "ai"
    / "datasets"
)

        self.SAMPLE_RATE = 16000

        self.DEVICE = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )


settings = Settings()