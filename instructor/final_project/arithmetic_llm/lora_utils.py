"""Utility functions for LoRA parameter efficiency and adapter management."""

from typing import Dict, Any

import torch

from .lora_config import LoRAConfig
from .transformer_model import ArithmeticTransformer


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """Count parameters in a model.

    Args:
        model: Model whose parameters to count
        trainable_only: If True, count only parameters with requires_grad=True

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_parameter_stats(model: torch.nn.Module) -> Dict[str, float]:
    """Get parameter counts and percentages.

    Returns:
        Dictionary with total, trainable, frozen, trainable_pct, frozen_pct
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    frozen_params = total_params - trainable_params
    trainable_pct = (trainable_params / total_params) * 100 if total_params else 0.0
    frozen_pct = (frozen_params / total_params) * 100 if total_params else 0.0

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
        "trainable_pct": trainable_pct,
        "frozen_pct": frozen_pct,
    }


def merge_lora_checkpoint(
    base_checkpoint_path: str,
    adapter_path: str,
    output_path: str
) -> str:
    """Merge a base checkpoint with a LoRA adapter and save merged model."""
    base_checkpoint = torch.load(base_checkpoint_path, map_location="cpu")
    model_state = base_checkpoint.get("model_state_dict")
    if model_state is None:
        raise ValueError("Base checkpoint missing model_state_dict")

    base_config = base_checkpoint.get("config", {})
    model_config = base_checkpoint.get("model_config")
    if model_config is None:
        model_config = {
            "vocab_size": base_checkpoint.get("tokenizer_vocab_size", 0),
            "d_model": base_config.get("d_model", 256),
            "nhead": base_config.get("nhead", 8),
            "num_layers": base_config.get("num_layers", 6),
            "dim_feedforward": base_config.get("dim_feedforward", 1024),
            "dropout": base_config.get("dropout", 0.1),
            "max_seq_length": base_config.get("max_seq_length", 512),
        }

    if model_config.get("vocab_size", 0) == 0:
        raise ValueError("Unable to infer vocab_size from base checkpoint")

    model = ArithmeticTransformer(**model_config)
    model.load_state_dict(model_state)

    adapter_data = torch.load(adapter_path, map_location="cpu")
    metadata = adapter_data.get("metadata")
    if metadata is None:
        raise ValueError("Adapter missing metadata")

    lora_config = LoRAConfig(
        rank=metadata["rank"],
        alpha=metadata["alpha"],
        target_modules=metadata["target_modules"],
        dropout=metadata.get("dropout", 0.0),
    )
    model.inject_lora(lora_config)
    model.load_lora_adapters(adapter_path)
    model.merge_lora_weights()

    merged_checkpoint: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "config": base_config,
        "model_config": model_config,
        "tokenizer_vocab_size": base_checkpoint.get("tokenizer_vocab_size", 0),
        "merged": True,
        "base_checkpoint": base_checkpoint_path,
        "adapter_path": adapter_path,
        "lora_metadata": metadata,
    }
    torch.save(merged_checkpoint, output_path)
    return output_path


