#!/usr/bin/env python3
"""Diagnose training speed issues."""

import torch
import time
from .transformer_model import ArithmeticTransformer
from .arithmetic_tokenizer import ArithmeticBPETokenizer

print("=" * 60)
print("TRAINING SPEED DIAGNOSTIC")
print("=" * 60)

# Check device (priority: CUDA > MPS > CPU)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"\nDevice: {device}")

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = ArithmeticBPETokenizer()
tokenizer.load("data/tokenizer")
vocab_size = len(tokenizer.token2id)
print(f"Vocabulary size: {vocab_size}")

# Create model
print("\nCreating model...")
model = ArithmeticTransformer(
    vocab_size=vocab_size,
    d_model=256,
    nhead=8,
    num_layers=6,
    dim_feedforward=1024,
    dropout=0.1,
    max_seq_length=512
)
model = model.to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test different batch sizes and sequence lengths
batch_sizes = [8, 16, 32, 64, 128]
seq_lengths = [512]

print("\n" + "=" * 60)
print("SPEED TEST")
print("=" * 60)

for batch_size in batch_sizes:
    for seq_length in seq_lengths:
        print(f"\nBatch size: {batch_size}, Sequence length: {seq_length}")
        
        # Create dummy batch
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
        attention_mask = torch.ones(batch_size, seq_length, device=device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
        
        # Warmup
        model.train()
        for _ in range(2):
            inputs = input_ids[:, :-1]
            targets = labels[:, 1:]
            input_attention_mask = attention_mask[:, :-1]
            logits = model(inputs, input_attention_mask)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100
            )
            loss.backward()
        
        # Time forward + backward pass
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        start = time.time()
        
        num_iterations = 5
        for _ in range(num_iterations):
            inputs = input_ids[:, :-1]
            targets = labels[:, 1:]
            input_attention_mask = attention_mask[:, :-1]
            
            logits = model(inputs, input_attention_mask)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100
            )
            loss.backward()
        
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        end = time.time()
        
        avg_time = (end - start) / num_iterations
        print(f"  Time per iteration: {avg_time:.3f}s")
        print(f"  Estimated epoch time (1407 batches): {avg_time * 1407 / 60:.1f} minutes")

print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)

print("\nIf all batch sizes are slow (>5s/iteration):")
print("  - MPS might have compatibility issues with this model")
print("  - Try: --device cpu (might actually be faster!)")
print("  - Or reduce model size: --d-model 128 --num-layers 4")

print("\nIf larger batches are much slower:")
print("  - Use smaller batch size (16 or 32)")
print("  - MPS memory bandwidth might be saturated")


