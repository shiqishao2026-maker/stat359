#!/usr/bin/env python3
"""Command-line interface for interactive arithmetic solving."""

import argparse
from .interactive_solver import InteractiveArithmeticSolver


def main():
    """Run interactive arithmetic solver from command line."""
    parser = argparse.ArgumentParser(
        description="Interactive arithmetic problem solver using trained LLM"
    )
    
    # Required arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to instruction-tuned model checkpoint"
    )
    
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path to tokenizer directory"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference: 'cuda', 'mps', 'cpu', or 'auto' (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        import torch
        device = (
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )
    else:
        device = args.device
    
    print("\n" + "=" * 60)
    print("INTERACTIVE ARITHMETIC SOLVER")
    print("=" * 60)
    print(f"\nModel: {args.model_path}")
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Create and run interactive solver
    try:
        solver = InteractiveArithmeticSolver(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            device=device
        )
        solver.run()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        print("Thank you for using Arithmetic LLM!")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("ERROR!")
        print("=" * 60)
        print(f"Failed to start interactive solver: {str(e)}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()


