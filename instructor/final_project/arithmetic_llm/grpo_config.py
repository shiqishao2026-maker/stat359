"""Configuration module for GRPO training."""

from dataclasses import dataclass
from typing import Any, Dict

import torch


@dataclass
class GRPOConfig:
    """Configuration for GRPO training.

    Attributes:
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        warmup_steps: Number of warmup steps for learning rate scheduler
        gradient_clip: Maximum gradient norm for clipping
        save_every: Save checkpoint every N steps
        eval_every: Evaluate model every N steps
        device: Device for training ('cuda', 'mps', or 'cpu')
        num_candidates: Number of candidates per prompt
        temperature: Sampling temperature
        top_k: Top-k sampling cutoff
        top_p: Top-p (nucleus) sampling cutoff
        kl_penalty_coef: KL divergence penalty coefficient
        advantage_epsilon: Epsilon for advantage normalization
        max_gen_length: Maximum generation length
        gradient_accumulation_steps: Steps for gradient accumulation
        log_every: Print training progress every N steps
    """

    learning_rate: float = 1e-5
    batch_size: int = 8
    num_epochs: int = 3
    warmup_steps: int = 100
    gradient_clip: float = 1.0
    save_every: int = 500
    eval_every: int = 250
    device: str = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    num_candidates: int = 4
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    kl_penalty_coef: float = 0.05
    advantage_epsilon: float = 1e-8
    max_gen_length: int = 256

    gradient_accumulation_steps: int = 1
    log_every: int = 50

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid
        """
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )

        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive, got {self.batch_size}"
            )

        if self.num_epochs <= 0:
            raise ValueError(
                f"num_epochs must be positive, got {self.num_epochs}"
            )

        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be non-negative, got {self.warmup_steps}"
            )

        if self.gradient_clip <= 0:
            raise ValueError(
                f"gradient_clip must be positive, got {self.gradient_clip}"
            )

        if self.save_every <= 0:
            raise ValueError(
                f"save_every must be positive, got {self.save_every}"
            )

        if self.eval_every <= 0:
            raise ValueError(
                f"eval_every must be positive, got {self.eval_every}"
            )

        if self.device not in ["cuda", "mps", "cpu"]:
            raise ValueError(
                f"device must be 'cuda', 'mps', or 'cpu', got {self.device}"
            )

        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "device is set to 'cuda' but CUDA is not available"
            )

        if self.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError(
                "device is set to 'mps' but MPS is not available"
            )

        if self.num_candidates < 2:
            raise ValueError(
                f"num_candidates must be at least 2, got {self.num_candidates}"
            )

        if self.temperature <= 0:
            raise ValueError(
                f"temperature must be positive, got {self.temperature}"
            )

        if self.top_k <= 0:
            raise ValueError(
                f"top_k must be positive, got {self.top_k}"
            )

        if self.top_p <= 0 or self.top_p > 1:
            raise ValueError(
                f"top_p must be in (0, 1], got {self.top_p}"
            )

        if self.kl_penalty_coef < 0:
            raise ValueError(
                f"kl_penalty_coef must be non-negative, got {self.kl_penalty_coef}"
            )

        if self.advantage_epsilon <= 0:
            raise ValueError(
                f"advantage_epsilon must be positive, got {self.advantage_epsilon}"
            )

        if self.max_gen_length <= 0:
            raise ValueError(
                f"max_gen_length must be positive, got {self.max_gen_length}"
            )

        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps must be positive, got "
                f"{self.gradient_accumulation_steps}"
            )
        if self.log_every <= 0:
            raise ValueError(
                f"log_every must be positive, got {self.log_every}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "warmup_steps": self.warmup_steps,
            "gradient_clip": self.gradient_clip,
            "save_every": self.save_every,
            "eval_every": self.eval_every,
            "device": self.device,
            "num_candidates": self.num_candidates,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "kl_penalty_coef": self.kl_penalty_coef,
            "advantage_epsilon": self.advantage_epsilon,
            "max_gen_length": self.max_gen_length,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "log_every": self.log_every,
        }
