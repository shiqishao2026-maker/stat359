#!/usr/bin/env python3
"""Command-line interface for instruction fine-tuning."""

import argparse
import json
from .train_instruction import train_instruction_model
from .training_config import TrainingConfig


def main():
    """Fine-tune instruction model from command line."""
    parser = argparse.ArgumentParser(
        description="Fine-tune arithmetic LLM with instruction formatting"
    )
    
    # Required arguments
    parser.add_argument(
        "--instruction-corpus-path",
        type=str,
        required=True,
        help="Path to instruction-formatted corpus file"
    )
    
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path to trained tokenizer directory"
    )
    
    parser.add_argument(
        "--foundational-checkpoint",
        type=str,
        required=True,
        help="Path to foundational model checkpoint"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save model checkpoints (default: homeowrk/arithmetic_llm/models)"
    )
    
    # Training configuration
    parser.add_argument(
        "--config",
        type=str,
        help="Path to training configuration JSON file"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5, lower than foundational)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="Number of fine-tuning epochs (default: 5)"
    )
    
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Number of warmup steps (default: 500)"
    )
    
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Gradient clipping value (default: 1.0)"
    )
    
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Save checkpoint every N steps (default: 500)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for training: 'cuda', 'mps', 'cpu', or 'auto' (default: auto)"
    )
    
    # Model configuration
    parser.add_argument(
        "--model-config",
        type=str,
        help="Path to model configuration JSON file (optional, will use foundational config if not provided)"
    )
    
    args = parser.parse_args()
    
    # Load or create training configuration
    if args.config:
        print(f"Loading training configuration from: {args.config}")
        config = TrainingConfig.from_json(args.config)
    else:
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
        
        config = TrainingConfig(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            warmup_steps=args.warmup_steps,
            gradient_clip=args.gradient_clip,
            save_every=args.save_every,
            device=device
        )
    
    # Load model configuration if provided
    model_config = None
    if args.model_config:
        print(f"Loading model configuration from: {args.model_config}")
        with open(args.model_config, 'r') as f:
            model_config = json.load(f)
    
    # Display configuration
    print("\n" + "=" * 60)
    print("INSTRUCTION FINE-TUNING")
    print("=" * 60)
    print(f"\nInstruction corpus: {args.instruction_corpus_path}")
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"Foundational checkpoint: {args.foundational_checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print("\nTraining Configuration:")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Warmup steps: {config.warmup_steps}")
    print(f"  Gradient clip: {config.gradient_clip}")
    print(f"  Save every: {config.save_every} steps")
    print(f"  Device: {config.device}")
    print("=" * 60 + "\n")
    
    # Train model
    try:
        final_checkpoint = train_instruction_model(
            instruction_corpus_path=args.instruction_corpus_path,
            tokenizer_path=args.tokenizer_path,
            foundational_checkpoint=args.foundational_checkpoint,
            output_dir=args.output_dir,
            config=config,
            model_config=model_config
        )
        
        print("\n" + "=" * 60)
        print("FINE-TUNING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Final checkpoint: {final_checkpoint}")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("FINE-TUNING FAILED!")
        print("=" * 60)
        print(f"Error: {str(e)}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()


