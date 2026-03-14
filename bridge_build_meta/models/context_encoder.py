from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class BeliefEncoderOutput:
    context: torch.Tensor
    hidden: torch.Tensor
    task_logits: torch.Tensor


class BeliefEncoder(nn.Module):
    """Encode structured bridge_build evidence into a recurrent latent belief."""

    def __init__(
        self,
        *,
        grid_channels: int,
        scalar_dim: int,
        hidden_dim: int,
        task_vocab_size: int,
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
        self.pre_gru = nn.Sequential(
            nn.Linear(cnn_channels + scalar_hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.task_head = nn.Linear(hidden_dim, task_vocab_size)
        self._init_parameters()

    def _init_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def initial_hidden(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

    def forward_step(
        self,
        grid: torch.Tensor,
        scalars: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> BeliefEncoderOutput:
        if grid.dim() != 4:
            raise ValueError("grid must have shape [B, C, H, W]")
        if scalars.dim() != 2:
            raise ValueError("scalars must have shape [B, S]")

        batch_size = grid.size(0)
        flat_grid = self.grid_tower(grid).flatten(start_dim=1)
        flat_scalars = self.scalar_tower(scalars)
        fused = self.pre_gru(torch.cat([flat_grid, flat_scalars], dim=-1))
        if hidden is None:
            hidden = self.initial_hidden(
                batch_size,
                device=fused.device,
                dtype=fused.dtype,
            )
        next_hidden = self.gru(fused, hidden)
        task_logits = self.task_head(next_hidden)
        return BeliefEncoderOutput(
            context=next_hidden,
            hidden=next_hidden,
            task_logits=task_logits,
        )
