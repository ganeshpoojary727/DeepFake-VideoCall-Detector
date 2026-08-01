"""Early stopping monitor module."""

from __future__ import annotations


class EarlyStopping:
    """Monitors validation loss to trigger early training termination on plateau."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower().strip()
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """Update monitor status with current metric score.

        Args:
            val_loss: Validation loss metric score.

        Returns:
            bool: True if training should stop.
        """
        score = -val_loss if self.mode == "min" else val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.should_stop = False

        return self.should_stop

    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False
