from __future__ import annotations

import torch
from torch import nn


class DiscardPolicyNet(nn.Module):
    """Predict discard tile index (0-33) from state features."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 34):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
