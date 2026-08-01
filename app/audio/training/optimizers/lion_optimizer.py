"""Lion (EvoLved Sign Momentum) optimizer PyTorch implementation."""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Tuple
import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    """Lion optimizer implementation in PyTorch.

    Reference: Symbolic Discovery of Optimization Algorithms (Chen et al., 2023)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Weight decay step
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # Update momentum and parameter using sign operator
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                p.add_(torch.sign(update), alpha=-group["lr"])

                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
