"""Training configuration module for arithmetic LLM training."""

import json
import torch
from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict

from .lora_config import LoRAConfig


@dataclass
class TrainingConfig:
    """Configuration for model training.
    
    Attributes:
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        warmup_steps: Number of warmup steps for learning rate scheduler
        gradient_clip: Maximum gradient norm for clipping
        save_every: Save checkpoint every N steps
        eval_every: Evaluate model every N steps
        device: Device for training ('cuda', 'mps', or 'cpu')
        lora_config: Optional LoRA configuration
    """
    
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    save_every: int = 1000
    eval_every: int = 500
    device: str = (
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    lora_config: Optional[LoRAConfig] = None
    
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

        if self.lora_config is not None:
            self.lora_config.validate()
    
    @classmethod
    def from_json(cls, json_path: str) -> 'TrainingConfig':
        """Load configuration from JSON file.
        
        Args:
            json_path: Path to JSON configuration file
            
        Returns:
            TrainingConfig instance
            
        Raises:
            FileNotFoundError: If JSON file doesn't exist
            ValueError: If JSON is malformed or contains invalid values
        """
        try:
            with open(json_path, 'r') as f:
                config_dict = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file not found: {json_path}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in configuration file: {e}"
            )
        
        # Create config instance
        lora_config = None
        if "lora_config" in config_dict and config_dict["lora_config"] is not None:
            lora_config = LoRAConfig(**config_dict["lora_config"])
        config_dict["lora_config"] = lora_config

        config = cls(**config_dict)
        
        # Validate configuration
        config.validate()
        
        return config
    
    def to_json(self, json_path: str) -> None:
        """Save configuration to JSON file.
        
        Args:
            json_path: Path to save JSON configuration file
        """
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        config_dict: Dict[str, Any] = asdict(self)
        if self.lora_config is not None:
            config_dict["lora_config"] = asdict(self.lora_config)
        return config_dict


