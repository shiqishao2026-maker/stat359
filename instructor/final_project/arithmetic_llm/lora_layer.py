"""LoRA layer implementation."""

import math
from typing import Optional

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """Linear layer wrapper with LoRA adapters."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(
                "base_layer must be an instance of torch.nn.Linear"
            )
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise ValueError(
                f"rank must be an integer, got {type(rank).__name__}"
            )
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            raise ValueError(
                f"alpha must be a number, got {type(alpha).__name__}"
            )
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool):
            raise ValueError(
                f"dropout must be a number, got {type(dropout).__name__}"
            )
        if dropout < 0 or dropout >= 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = float(alpha)
        self.dropout = float(dropout)
        self.scaling = self.alpha / self.rank

        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(
            torch.empty(rank, self.base_layer.in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.base_layer.out_features, rank)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout: Optional[nn.Dropout] = None
        if self.dropout > 0:
            self.lora_dropout = nn.Dropout(p=self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation."""
        base_output = self.base_layer(x)

        lora_input = x
        if self.lora_dropout is not None:
            lora_input = self.lora_dropout(lora_input)

        lora_update = torch.matmul(lora_input, self.lora_A.t())
        lora_update = torch.matmul(lora_update, self.lora_B.t())
        lora_update = lora_update * self.scaling

        return base_output + lora_update

    def lora_parameters(self):
        """Return LoRA trainable parameters only."""
        return [self.lora_A, self.lora_B]
