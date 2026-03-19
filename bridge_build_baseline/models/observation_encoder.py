from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ObservationEncoderOutput:
    context: torch.Tensor


class ObservationEncoder(nn.Module):
    """Encode the current structured observation without a recurrent belief state."""

    def __init__(
        self,
        *,
        grid_channels: int,
        scalar_dim: int,
        hidden_dim: int,
        cnn_channels: int = 32,
        scalar_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.grid_tower = nn.Sequential(
            nn.Conv2d(grid_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.scalar_tower = nn.Sequential(
            nn.Linear(scalar_dim, scalar_hidden_dim),
            nn.ReLU(),
            nn.Linear(scalar_hidden_dim, scalar_hidden_dim),
            nn.ReLU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(cnn_channels + scalar_hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self._init_parameters()

    def _init_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_step(
        self,
        *,
        grid: torch.Tensor,
        scalars: torch.Tensor,
    ) -> ObservationEncoderOutput:
        if grid.dim() != 4:
            raise ValueError("grid must have shape [B, C, H, W]")
        if scalars.dim() != 2:
            raise ValueError("scalars must have shape [B, S]")
        flat_grid = self.grid_tower(grid).flatten(start_dim=1)
        flat_scalars = self.scalar_tower(scalars)
        context = self.proj(torch.cat([flat_grid, flat_scalars], dim=-1))
        return ObservationEncoderOutput(context=context)
