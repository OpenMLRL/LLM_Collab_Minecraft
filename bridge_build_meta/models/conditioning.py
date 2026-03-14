from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class ContextTokenBias(nn.Module):
    """Project a latent belief vector to a token-embedding bias."""

    def __init__(self, context_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )
        self._init_parameters()

    def _init_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, context_vec: torch.Tensor) -> torch.Tensor:
        return self.net(context_vec)


class ContextValueProjection(nn.Module):
    """Project latent context before concatenating with critic hidden states."""

    def __init__(self, context_dim: int, proj_dim: Optional[int] = None) -> None:
        super().__init__()
        out_dim = int(proj_dim or context_dim)
        self.output_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(context_dim, out_dim),
            nn.Tanh(),
        )
        self._init_parameters()

    def _init_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, context_vec: torch.Tensor) -> torch.Tensor:
        return self.net(context_vec)
