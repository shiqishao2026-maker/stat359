import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyStoriesConfig:
    """Configuration class for TinyStories model parameters."""
    
    def __init__(
        self,
        vocab_size=10000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=1024,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        window_size=256,  # NEW: attention window size
    ):
        """Initialize the TinyStories configuration.
        
        Args:
            vocab_size (int): Vocabulary size of the model
            hidden_size (int): Size of the hidden layers
            num_hidden_layers (int): Number of transformer layers
            num_attention_heads (int): Number of attention heads
            intermediate_size (int): Size of the intermediate (feed-forward) layer
            hidden_dropout_prob (float): Dropout probability for hidden layers
            attention_probs_dropout_prob (float): Dropout probability for attention probabilities
            max_position_embeddings (int): Maximum sequence length
            initializer_range (float): Range for weight initialization
            layer_norm_eps (float): Epsilon for layer normalization
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.window_size = window_size


class TinyStoriesEmbeddings(nn.Module):
    """Embeddings module for TinyStories model.
    
    Includes token embeddings and positional embeddings.
    """
    
    def __init__(self, config):
        """Initialize the embeddings.
        
        Args:
            config: TinyStoriesConfig instance
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        
    def forward(self, input_ids):
        """Forward pass for embeddings.
        
        Args:
            input_ids: Tensor of shape (batch_size, sequence_length) containing token IDs
            
        Returns:
            Tensor of shape (batch_size, sequence_length, hidden_size) containing embeddings
        """
        seq_length = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_length]
        
        # Get token embeddings
        inputs_embeds = self.word_embeddings(input_ids)
        
        # Get position embeddings
        position_embeddings = self.position_embeddings(position_ids)
        
        # Add token and position embeddings
        embeddings = inputs_embeds + position_embeddings
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        return embeddings


class TinyStoriesSelfAttention(nn.Module):
    """Self-attention layer for TinyStories model."""
    
    def __init__(self, config):
        """Initialize the self-attention layer.
        
        Args:
            config: TinyStoriesConfig instance
        """
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query, key, value projections
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Output projection
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x):
        """Reshape the tensor for multi-head attention.
        
        Args:
            x: Tensor of shape (batch_size, seq_length, all_head_size)
            
        Returns:
            Tensor of shape (batch_size, num_attention_heads, seq_length, attention_head_size)
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        """Forward pass for self-attention.
        
        Args:
            hidden_states: Tensor of shape (batch_size, seq_length, hidden_size)
            attention_mask: Optional mask of shape (batch_size, 1, 1, seq_length)
            
        Returns:
            Tuple containing:
                - attention_output: Tensor of shape (batch_size, seq_length, hidden_size)
                - attention_probs: Tensor of shape (batch_size, num_attention_heads, seq_length, seq_length)
        """
        # Project inputs to queries, keys, and values
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Take the dot product between "query" and "key" to get the raw attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Normalize the attention scores to probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply dropout to attention probabilities
        attention_probs = self.dropout(attention_probs)
        
        # Calculate the context vector
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        # Reshape back to (batch_size, seq_length, hidden_size)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Apply output projection
        attention_output = self.output(context_layer)
        
        return attention_output, attention_probs


class TinyStoriesFeedForward(nn.Module):
    """Feed-forward layer for TinyStories model."""
    
    def __init__(self, config):
        """Initialize the feed-forward layer.
        
        Args:
            config: TinyStoriesConfig instance
        """
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states):
        """Forward pass for feed-forward layer.
        
        Args:
            hidden_states: Tensor of shape (batch_size, seq_length, hidden_size)
            
        Returns:
            Tensor of shape (batch_size, seq_length, hidden_size)
        """
        hidden_states = self.dense1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states


class TinyStoriesLayer(nn.Module):
    """Transformer layer for TinyStories model."""
    
    def __init__(self, config):
        """Initialize the transformer layer.
        
        Args:
            config: TinyStoriesConfig instance
        """
        super().__init__()
        self.attention = TinyStoriesSelfAttention(config)
        self.feed_forward = TinyStoriesFeedForward(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states, attention_mask=None):
        """Forward pass for transformer layer.
        
        Args:
            hidden_states: Tensor of shape (batch_size, seq_length, hidden_size)
            attention_mask: Optional mask of shape (batch_size, 1, 1, seq_length)
            
        Returns:
            Tuple containing:
                - hidden_states: Tensor of shape (batch_size, seq_length, hidden_size)
                - attention_probs: Tensor of shape (batch_size, num_attention_heads, seq_length, seq_length)
        """
        # Self-attention with residual connection and layer normalization
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)
        hidden_states = residual + attention_output
        
        # Feed-forward with residual connection and layer normalization
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = residual + feed_forward_output
        
        return hidden_states, attention_probs


class TinyStoriesModel(nn.Module):
    """TinyStories transformer model."""
    
    def __init__(self, config):
        """Initialize the TinyStories model.
        
        Args:
            config: TinyStoriesConfig instance
        """
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embeddings = TinyStoriesEmbeddings(config)
        
        # Transformer layers
        self.layers = nn.ModuleList([TinyStoriesLayer(config) for _ in range(config.num_hidden_layers)])
        
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights.
        
        Args:
            module: PyTorch module to initialize
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_input_embeddings(self):
        """Get the input embeddings.
        
        Returns:
            nn.Embedding: The input embeddings
        """
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        """Set the input embeddings.
        
        Args:
            value: New embeddings to set
        """
        self.embeddings.word_embeddings = value
    
    def _get_causal_mask(self, seq_length, device, window_size=None):
        """Create a local causal (look-ahead) mask for self-attention with a limited window size."""
        if window_size is None:
            window_size = self.config.window_size if hasattr(self, 'config') and hasattr(self.config, 'window_size') else 256
        mask = torch.full((seq_length, seq_length), float('-inf'), device=device)
        for i in range(seq_length):
            start = max(0, i - window_size + 1)
            mask[i, start:i+1] = 0.0
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        output_attentions=False,
    ):
        """Forward pass for TinyStories model.
        
        Args:
            input_ids: Optional tensor of shape (batch_size, seq_length) containing token IDs
            attention_mask: Optional tensor of shape (batch_size, seq_length) containing attention mask
            inputs_embeds: Optional tensor of shape (batch_size, seq_length, hidden_size) containing embeddings
            output_attentions: Whether to return attention probabilities
            
        Returns:
            Dict containing:
                - last_hidden_state: Tensor of shape (batch_size, seq_length, hidden_size)
                - attentions: Optional list of attention probabilities
        """
        # Get embeddings
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            hidden_states = self.embeddings(input_ids)
            seq_length = input_ids.size(1)
            device = input_ids.device
        elif inputs_embeds is not None:
            hidden_states = inputs_embeds
            seq_length = inputs_embeds.size(1)
            device = inputs_embeds.device
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        # Prepare causal mask
        causal_mask = self._get_causal_mask(seq_length, device, window_size=self.config.window_size)

        # Prepare attention mask (padding)
        if attention_mask is not None:
            # Create a 4D attention mask (batch_size, 1, 1, seq_length)
            attention_mask_ = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask_ = (1.0 - attention_mask_) * -10000.0
            # Combine causal and padding masks (broadcast causal mask to batch size)
            combined_mask = causal_mask + attention_mask_
        else:
            combined_mask = causal_mask

        # Process through transformer layers
        all_attentions = [] if output_attentions else None
        
        for layer in self.layers:
            hidden_states, attention_probs = layer(hidden_states, combined_mask)
            
            if output_attentions:
                all_attentions.append(attention_probs)
        
        # Apply final layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        # Prepare output
        output = {
            "last_hidden_state": hidden_states,
        }
        
        if output_attentions:
            output["attentions"] = all_attentions
        
        return output


class TinyStoriesForCausalLM(nn.Module):
    """TinyStories model for causal language modeling."""
    
    def __init__(self, config):
        """Initialize the TinyStories model for causal language modeling.
        
        Args:
            config: TinyStoriesConfig instance
        """
        super().__init__()
        self.config = config
        
        # Base transformer model
        self.transformer = TinyStoriesModel(config)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie weights between input embeddings and output embeddings
        self.tie_weights()
        
    def _init_weights(self, module):
        """Initialize the weights.
        
        Args:
            module: PyTorch module to initialize
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def tie_weights(self):
        """Tie the weights between the input embeddings and the output embeddings."""
        self.lm_head.weight = self.transformer.get_input_embeddings().weight
    
    def get_output_embeddings(self):
        """Get the output embeddings.
        
        Returns:
            nn.Linear: The output embeddings
        """
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        """Set the output embeddings.
        
        Args:
            new_embeddings: New embeddings to set
        """
        self.lm_head = new_embeddings
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=False,
    ):
        """Forward pass for causal language modeling.
        
        Args:
            input_ids: Optional tensor of shape (batch_size, seq_length) containing token IDs
            attention_mask: Optional tensor of shape (batch_size, seq_length) containing attention mask
            inputs_embeds: Optional tensor of shape (batch_size, seq_length, hidden_size) containing embeddings
            labels: Optional tensor of shape (batch_size, seq_length) containing labels for language modeling
            output_attentions: Whether to return attention probabilities
            
        Returns:
            Dict containing:
                - loss: Optional language modeling loss
                - logits: Tensor of shape (batch_size, seq_length, vocab_size) containing logits
                - attentions: Optional list of attention probabilities
        """
        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )
        
        hidden_states = transformer_outputs["last_hidden_state"]
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        # Prepare output
        output = {
            "logits": logits,
        }
        
        # Calculate loss if labels are provided
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
            output["loss"] = loss
        
        if output_attentions and "attentions" in transformer_outputs:
            output["attentions"] = transformer_outputs["attentions"]
        
        return output
    
    def generate(self, input_ids, max_length, num_beams=1, temperature=1.0, top_k=0, top_p=0.9, eos_token_id=None):
        """Generate text using the model.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_length) containing token IDs
            max_length: Maximum length of the generated sequence
            num_beams: Number of beams for beam search
            temperature: Temperature for sampling
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            eos_token_id: Optional; if set, generation stops when this token is produced
        Returns:
            Tensor of shape (batch_size, max_length) containing generated token IDs
        """
        batch_size = input_ids.shape[0]
        
        # If num_beams > 1, use beam search
        if num_beams > 1:
            return self._generate_beam_search(
                input_ids, max_length, num_beams, temperature, top_k, top_p, eos_token_id
            )
        
        # Otherwise, use greedy or sampling
        cur_len = input_ids.shape[1]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        while cur_len < max_length:
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(input_ids=input_ids)
                next_token_logits = outputs["logits"][:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float("Inf")
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float("Inf")
            
            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # If eos_token_id is set, mark finished sequences
            if eos_token_id is not None:
                next_token = next_token.masked_fill(finished, eos_token_id)
                finished = finished | (next_token == eos_token_id)
            
            # Add the new token to the input
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
            cur_len += 1
            if eos_token_id is not None and finished.all():
                break
        return input_ids
    
    def _generate_beam_search(self, input_ids, max_length, num_beams, temperature, top_k, top_p, eos_token_id=None):
        """Generate text using beam search.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_length) containing token IDs
            max_length: Maximum length of the generated sequence
            num_beams: Number of beams for beam search
            temperature: Temperature for sampling
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            eos_token_id: Optional; if set, generation stops when this token is produced
        Returns:
            Tensor of shape (batch_size, max_length) containing generated token IDs
        """
        batch_size = input_ids.shape[0]
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, -1)
        input_ids = input_ids.contiguous().view(batch_size * num_beams, -1)
        beam_scores = torch.zeros((batch_size, num_beams), device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)
        generated = input_ids.clone()
        finished = torch.zeros(batch_size * num_beams, dtype=torch.bool, device=input_ids.device)
        for i in range(max_length - input_ids.shape[1]):
            with torch.no_grad():
                outputs = self.forward(input_ids=generated)
                next_token_logits = outputs["logits"][:, -1, :]
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float("Inf")
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float("Inf")
            next_scores = F.log_softmax(next_token_logits, dim=-1)
            next_scores = next_scores + beam_scores[:, None]
            next_scores = next_scores.view(batch_size, num_beams * self.config.vocab_size)
            topk_scores, topk_tokens = torch.topk(next_scores, num_beams, dim=1)
            topk_beam_indices = topk_tokens // self.config.vocab_size
            topk_token_indices = topk_tokens % self.config.vocab_size
            topk_beam_indices = topk_beam_indices.view(-1)
            topk_token_indices = topk_token_indices.view(-1, 1)
            beam_scores = topk_scores.view(-1)
            beam_indices = (
                torch.arange(batch_size * num_beams, device=input_ids.device) // num_beams
            ) * num_beams + topk_beam_indices
            generated = generated[beam_indices]
            # If eos_token_id is set, mark finished beams
            if eos_token_id is not None:
                topk_token_indices = topk_token_indices.masked_fill(finished.unsqueeze(1), eos_token_id)
                finished = finished | (topk_token_indices.squeeze(1) == eos_token_id)
            generated = torch.cat([generated, topk_token_indices], dim=1)
            if eos_token_id is not None and finished.all():
                break
        final_generated = []
        for i in range(batch_size):
            final_generated.append(generated[i * num_beams])
        return torch.stack(final_generated)
