#!/usr/bin/env python3
"""Command-line interface for tokenizer training."""

import argparse
import os
from .arithmetic_tokenizer import ArithmeticBPETokenizer


def main():
    """Train BPE tokenizer from command line."""
    parser = argparse.ArgumentParser(
        description="Train BPE tokenizer on arithmetic corpus"
    )
    
    parser.add_argument(
        "--corpus-path",
        type=str,
        required=True,
        help="Path to training corpus file"
    )
    
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=1000,
        help="Target vocabulary size (default: 1000)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/tokenizer",
        help="Directory to save trained tokenizer (default: data/tokenizer)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.corpus_path):
        parser.error(f"Corpus file not found: {args.corpus_path}")
    
    if args.vocab_size <= 0:
        parser.error("vocab-size must be positive")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train tokenizer
    print(f"Training BPE tokenizer with vocabulary size {args.vocab_size}...")
    print(f"Corpus: {args.corpus_path}")
    
    tokenizer = ArithmeticBPETokenizer(vocab_size=args.vocab_size)
    tokenizer.train(args.corpus_path)
    
    print(f"Saving tokenizer to: {args.output_dir}")
    tokenizer.save(args.output_dir)
    
    # Display tokenizer statistics
    print("\nTokenizer Statistics:")
    print(f"  Vocabulary size: {len(tokenizer.token2id)}")
    print(f"  BPE merge operations: {len(tokenizer.bpe_codes)}")
    print(f"  Special tokens: {', '.join(tokenizer.special_tokens)}")
    
    # Test tokenizer with a sample expression
    for test_expr in ["5 + 10 - 3", "12 - (4 + 2)", "((7+3)-(3+5))"]:
        encoded = tokenizer.encode(test_expr, add_special_tokens=True)
        decoded = tokenizer.decode(encoded, skip_special_tokens=False)
        print("\nTest encoding:")
        print(f"  Input: {test_expr}")
        print(f"  Encoded (with BOS/EOS): {encoded}")
        print(f"  Decoded: {decoded}")
        
        # Test without special tokens
        encoded_no_special = tokenizer.encode(test_expr, add_special_tokens=False)
        print(f"  Encoded (without BOS/EOS): {encoded_no_special}")
    
    print("\nTokenizer training complete!")


if __name__ == "__main__":
    main()


