"""LoRA configuration module."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning.

    Attributes:
        rank: Rank of the LoRA decomposition matrices
        alpha: Scaling factor for LoRA updates
        target_modules: List of module names to apply LoRA
        dropout: Dropout probability on the LoRA path
    """

    rank: int = 8
    alpha: float = 16.0
    target_modules: List[str] = field(default_factory=lambda: ["attention"])
    dropout: float = 0.0

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid
        """
        if not isinstance(self.rank, int) or isinstance(self.rank, bool):
            raise ValueError(
                f"rank must be an integer, got {type(self.rank).__name__}"
            )
        if self.rank <= 0:
            raise ValueError(
                f"rank must be positive, got {self.rank}"
            )

        if not isinstance(self.alpha, (int, float)) or isinstance(self.alpha, bool):
            raise ValueError(
                f"alpha must be a number, got {type(self.alpha).__name__}"
            )
        if self.alpha < 0:
            raise ValueError(
                f"alpha must be non-negative, got {self.alpha}"
            )

        if not isinstance(self.dropout, (int, float)) or isinstance(self.dropout, bool):
            raise ValueError(
                f"dropout must be a number, got {type(self.dropout).__name__}"
            )
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError(
                f"dropout must be in [0, 1), got {self.dropout}"
            )

        if not isinstance(self.target_modules, list) or len(self.target_modules) == 0:
            raise ValueError(
                "target_modules must be a non-empty list of strings"
            )
        if not all(isinstance(module, str) and module for module in self.target_modules):
            raise ValueError(
                "target_modules must be a non-empty list of strings"
            )
