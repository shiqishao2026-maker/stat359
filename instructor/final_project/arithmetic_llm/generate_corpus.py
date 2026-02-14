#!/usr/bin/env python3
"""Command-line interface for corpus generation."""

import argparse
import os
from .corpus_generator import CorpusGenerator


def main():
    """Generate arithmetic training corpus from command line."""
    parser = argparse.ArgumentParser(
        description="Generate arithmetic training corpus for LLM training"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        required=True,
        help="Number of expression-evaluation pairs to generate"
    )
    
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Maximum depth of expression trees (default: 5)"
    )
    
    parser.add_argument(
        "--num-range",
        type=int,
        nargs=2,
        default=[1, 20],
        metavar=("MIN", "MAX"),
        help="Range of numbers to use in expressions (default: 1 20)"
    )
    
    parser.add_argument(
        "--invalid-rate",
        type=float,
        default=0.1,
        help="Fraction of invalid expressions to include (default: 0.1)"
    )
    
    parser.add_argument(
        "--output-foundational",
        type=str,
        default="data/foundational_corpus.txt",
        help="Path to save foundational corpus (default: homeowrk/arithmetic_llm/data/foundational_corpus.txt)"
    )
    
    parser.add_argument(
        "--output-instruction",
        type=str,
        default="data/instruction_corpus.txt",
        help="Path to save instruction corpus (default: homeowrk/arithmetic_llm/data/instruction_corpus.txt)"
    )
    
    parser.add_argument(
        "--foundational-only",
        action="store_true",
        help="Generate only foundational corpus"
    )
    
    parser.add_argument(
        "--instruction-only",
        action="store_true",
        help="Generate only instruction corpus"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.num_samples <= 0:
        parser.error("num-samples must be positive")
    
    if args.max_depth <= 0:
        parser.error("max-depth must be positive")
    
    if args.num_range[0] >= args.num_range[1]:
        parser.error("num-range MIN must be less than MAX")
    
    if not 0.0 <= args.invalid_rate <= 1.0:
        parser.error("invalid-rate must be between 0.0 and 1.0")
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(args.output_foundational), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_instruction), exist_ok=True)
    
    # Generate corpora
    num_range = tuple(args.num_range)
    
    if not args.instruction_only:
        print(f"Generating foundational corpus with {args.num_samples} samples...")
        generator = CorpusGenerator(
            num_samples=args.num_samples,
            max_depth=args.max_depth,
            num_range=num_range,
            invalid_rate=args.invalid_rate,
            output_path=args.output_foundational
        )
        generator.generate_corpus()
        print(f"Foundational corpus saved to: {args.output_foundational}")
    
    if not args.foundational_only:
        print(f"Generating instruction corpus with {args.num_samples} samples...")
        generator = CorpusGenerator(
            num_samples=args.num_samples,
            max_depth=args.max_depth,
            num_range=num_range,
            invalid_rate=args.invalid_rate,
            output_path=args.output_instruction
        )
        generator.generate_instruction_corpus(args.output_instruction)
        print(f"Instruction corpus saved to: {args.output_instruction}")
    
    print("Corpus generation complete!")


if __name__ == "__main__":
    main()


