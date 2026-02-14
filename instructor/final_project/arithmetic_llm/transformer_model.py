"""Transformer model for arithmetic reasoning.

This module implements a decoder-only transformer architecture adapted from the
TinyStories project for arithmetic expression evaluation with step-by-step reasoning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Iterator, Any

from .lora_config import LoRAConfig
from .lora_layer import LoRALayer


class ArithmeticTransformer(nn.Module):
    """Transformer model for arithmetic reasoning.
    
    A decoder-only transformer with causal masking for autoregressive generation
    of arithmetic solutions with step-by-step reasoning.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_length: int = 512,
    ):
        """Initialize the ArithmeticTransformer model.
        
        Args:
            vocab_size: Size of token vocabulary
            d_model: Dimension of model embeddings
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        self.register_buffer(
            "position_ids", torch.arange(max_seq_length).expand((1, -1))
        )
        
        # Dropout for embeddings
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights between input embeddings and output projection
        self.output_projection.weight = self.token_embedding.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
    
    def _create_causal_mask(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """Create causal (look-ahead) mask for autoregressive generation.
        
        Args:
            seq_length: Length of the sequence
            device: Device to create the mask on
            
        Returns:
            Causal mask of shape (seq_length, seq_length)
        """
        mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_length)
            attention_mask: Optional padding mask of shape (batch_size, seq_length)
                           where 1 indicates valid tokens and 0 indicates padding
        
        Returns:
            Logits of shape (batch_size, seq_length, vocab_size)
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)
        position_ids = self.position_ids[:, :seq_length]
        position_embeds = self.position_embedding(position_ids)
        
        # Combine token and position embeddings
        hidden_states = token_embeds + position_embeds
        hidden_states = self.embedding_dropout(hidden_states)
        
        # Create causal mask
        causal_mask = self._create_causal_mask(seq_length, device)
        
        # Combine causal mask with padding mask if provided
        if attention_mask is not None:
            # Convert padding mask to attention mask format
            # attention_mask: (batch_size, seq_length) with 1 for valid, 0 for padding
            # We need to create a mask of shape (batch_size, seq_length, seq_length)
            padding_mask = attention_mask.unsqueeze(1)  # (batch_size, 1, seq_length)
            padding_mask = (1.0 - padding_mask) * -10000.0
            
            # Broadcast causal mask to batch size and combine
            combined_mask = causal_mask.unsqueeze(0) + padding_mask
        else:
            combined_mask = causal_mask.unsqueeze(0)  # (1, seq_length, seq_length)
        
        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, combined_mask)
        
        # Apply final layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        # Project to vocabulary
        logits = self.output_projection(hidden_states)
        
        return logits

    def inject_lora(self, config: LoRAConfig) -> None:
        """Inject LoRA layers into attention and/or feedforward modules."""
        config.validate()
        target_modules = {module.lower() for module in config.target_modules}

        for layer in self.layers:
            if "attention" in target_modules:
                self._replace_linear_with_lora(layer.self_attention, "q_proj", config)
                self._replace_linear_with_lora(layer.self_attention, "k_proj", config)
                self._replace_linear_with_lora(layer.self_attention, "v_proj", config)
                self._replace_linear_with_lora(layer.self_attention, "out_proj", config)
            if "feedforward" in target_modules:
                self._replace_linear_with_lora(layer.feedforward, "linear1", config)
                self._replace_linear_with_lora(layer.feedforward, "linear2", config)

        self.lora_config = config

    def get_lora_parameters(self) -> Iterator[nn.Parameter]:
        """Return iterator over LoRA parameters only."""
        for module in self.modules():
            if isinstance(module, LoRALayer):
                for param in module.lora_parameters():
                    yield param

    def save_lora_adapters(self, path: str, base_model_path: Optional[str] = None) -> None:
        """Save LoRA adapter weights and metadata to a file."""
        lora_state: Dict[str, torch.Tensor] = {}
        for name, module in self.named_modules():
            if isinstance(module, LoRALayer):
                lora_state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
                lora_state[f"{name}.lora_B"] = module.lora_B.detach().cpu()

        if not lora_state:
            raise ValueError("No LoRA layers found to save")

        if not hasattr(self, "lora_config"):
            raise ValueError("LoRA configuration not set; call inject_lora first")

        metadata = {
            "rank": self.lora_config.rank,
            "alpha": self.lora_config.alpha,
            "target_modules": self.lora_config.target_modules,
            "dropout": self.lora_config.dropout,
            "base_model_path": base_model_path,
        }

        torch.save({"lora_state": lora_state, "metadata": metadata}, path)

    def load_lora_adapters(self, path: str) -> LoRAConfig:
        """Load LoRA adapter weights from a file."""
        data = torch.load(path, map_location="cpu")
        if "metadata" not in data or "lora_state" not in data:
            raise ValueError("Invalid LoRA adapter file format")

        metadata = data["metadata"]
        try:
            config = LoRAConfig(
                rank=metadata["rank"],
                alpha=metadata["alpha"],
                target_modules=metadata["target_modules"],
                dropout=metadata.get("dropout", 0.0),
            )
        except KeyError as exc:
            raise ValueError(f"Missing LoRA metadata field: {exc}") from exc

        config.validate()

        has_lora_layers = any(
            isinstance(module, LoRALayer) for module in self.modules()
        )
        if not has_lora_layers:
            self.inject_lora(config)

        if hasattr(self, "lora_config") and self.lora_config != config:
            raise ValueError("LoRA configuration does not match model")
        if not hasattr(self, "lora_config"):
            for module in self.modules():
                if isinstance(module, LoRALayer):
                    if module.rank != config.rank or module.alpha != config.alpha or module.dropout != config.dropout:
                        raise ValueError("LoRA configuration does not match model")
                    break
            self.lora_config = config

        lora_state: Dict[str, torch.Tensor] = data["lora_state"]
        device = next(self.parameters()).device

        matched_keys = set()
        for name, module in self.named_modules():
            if not isinstance(module, LoRALayer):
                continue
            key_a = f"{name}.lora_A"
            key_b = f"{name}.lora_B"
            if key_a not in lora_state or key_b not in lora_state:
                raise ValueError(f"Missing LoRA parameters for module {name}")
            module.lora_A.data.copy_(lora_state[key_a].to(device))
            module.lora_B.data.copy_(lora_state[key_b].to(device))
            matched_keys.update({key_a, key_b})

        extra_keys = set(lora_state.keys()) - matched_keys
        if extra_keys:
            raise ValueError("Adapter contains unexpected parameters")

        return config

    def merge_lora_weights(self) -> None:
        """Merge LoRA weights into base model weights and remove LoRA layers."""
        lora_module_names = [
            name for name, module in self.named_modules()
            if isinstance(module, LoRALayer)
        ]

        for name in lora_module_names:
            module = self._get_module_by_name(name)
            if not isinstance(module, LoRALayer):
                continue
            base_layer = module.base_layer
            merged_weight = base_layer.weight.detach().clone()
            lora_update = torch.matmul(module.lora_B, module.lora_A)
            merged_weight = merged_weight + lora_update * module.scaling

            new_linear = nn.Linear(
                base_layer.in_features,
                base_layer.out_features,
                bias=base_layer.bias is not None
            )
            new_linear = new_linear.to(
                device=base_layer.weight.device,
                dtype=base_layer.weight.dtype
            )
            new_linear.weight.data.copy_(merged_weight)
            if base_layer.bias is not None:
                new_linear.bias.data.copy_(base_layer.bias.detach())

            self._set_module_by_name(name, new_linear)

    def _replace_linear_with_lora(
        self,
        parent: nn.Module,
        attr_name: str,
        config: LoRAConfig
    ) -> None:
        layer = getattr(parent, attr_name)
        if isinstance(layer, LoRALayer):
            return
        if not isinstance(layer, nn.Linear):
            raise TypeError(
                f"Expected nn.Linear at {attr_name}, got {type(layer).__name__}"
            )
        setattr(
            parent,
            attr_name,
            LoRALayer(
                layer,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
            ),
        )

    def _get_module_by_name(self, module_name: str) -> nn.Module:
        parent, attr = self._resolve_module_parent(module_name)
        if attr.isdigit():
            return parent[int(attr)]
        return getattr(parent, attr)

    def _set_module_by_name(self, module_name: str, new_module: nn.Module) -> None:
        parent, attr = self._resolve_module_parent(module_name)
        if attr.isdigit():
            parent[int(attr)] = new_module
        else:
            setattr(parent, attr, new_module)

    def _resolve_module_parent(self, module_name: str) -> Tuple[nn.Module, str]:
        parts = module_name.split(".")
        parent: Any = self
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        return parent, parts[-1]
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 256,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate text autoregressively with temperature sampling.
        
        Args:
            input_ids: Initial token IDs of shape (batch_size, seq_length)
            max_length: Maximum length of generated sequence
            temperature: Temperature for sampling (higher = more random)
            top_k: If > 0, only sample from top k tokens
            top_p: Nucleus sampling threshold (only sample from top p probability mass)
            eos_token_id: Optional end-of-sequence token ID to stop generation
            attention_mask: Optional padding mask of shape (batch_size, seq_length)
        
        Returns:
            Generated token IDs of shape (batch_size, generated_length)
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Track which sequences are finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        generated = input_ids.clone()
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        with torch.no_grad():
            while generated.shape[1] < max_length:
                # Forward pass
                logits = self.forward(generated, attention_mask=attention_mask)
                
                # Get logits for the last token
                next_token_logits = logits[:, -1, :]
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep at least the first token
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter back to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                
                # Handle EOS token
                if eos_token_id is not None:
                    # Don't generate more tokens for finished sequences
                    next_token = next_token.masked_fill(finished, eos_token_id)
                    finished = finished | (next_token == eos_token_id)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((batch_size, 1), device=device)],
                        dim=-1
                    )
                
                # Stop if all sequences are finished
                if eos_token_id is not None and finished.all():
                    break
        
        return generated


class TransformerLayer(nn.Module):
    """Single transformer layer with self-attention and feedforward network."""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float
    ):
        """Initialize transformer layer.
        
        Args:
            d_model: Dimension of model embeddings
            nhead: Number of attention heads
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
        """
        super().__init__()
        
        # Self-attention
        self.self_attention = MultiHeadAttention(d_model, nhead, dropout)
        
        # Feedforward network
        self.feedforward = FeedForward(d_model, dim_feedforward, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through transformer layer.
        
        Args:
            hidden_states: Input of shape (batch_size, seq_length, d_model)
            attention_mask: Optional mask of shape (batch_size, seq_length, seq_length)
        
        Returns:
            Output of shape (batch_size, seq_length, d_model)
        """
        # Self-attention with residual connection and layer norm (pre-norm)
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.self_attention(hidden_states, attention_mask)
        hidden_states = self.dropout1(hidden_states)
        hidden_states = residual + hidden_states
        
        # Feedforward with residual connection and layer norm (pre-norm)
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.feedforward(hidden_states)
        hidden_states = self.dropout2(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, d_model: int, nhead: int, dropout: float):
        """Initialize multi-head attention.
        
        Args:
            d_model: Dimension of model embeddings
            nhead: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # Query, key, value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through multi-head attention.
        
        Args:
            hidden_states: Input of shape (batch_size, seq_length, d_model)
            attention_mask: Optional mask of shape (batch_size, seq_length, seq_length)
        
        Returns:
            Output of shape (batch_size, seq_length, d_model)
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project to queries, keys, values
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        # (batch_size, seq_length, d_model) -> (batch_size, nhead, seq_length, head_dim)
        q = q.view(batch_size, seq_length, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.nhead, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask.unsqueeze(1)
        
        # Compute attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, v)
        
        # Reshape back to (batch_size, seq_length, d_model)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_length, self.d_model)
        
        # Apply output projection
        output = self.out_proj(context)
        
        return output


class FeedForward(nn.Module):
    """Feedforward network with GELU activation."""
    
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float):
        """Initialize feedforward network.
        
        Args:
            d_model: Dimension of model embeddings
            dim_feedforward: Dimension of hidden layer
            dropout: Dropout rate
        """
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through feedforward network.
        
        Args:
            hidden_states: Input of shape (batch_size, seq_length, d_model)
        
        Returns:
            Output of shape (batch_size, seq_length, d_model)
        """
        hidden_states = self.linear1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear2(hidden_states)
        
        return hidden_states


