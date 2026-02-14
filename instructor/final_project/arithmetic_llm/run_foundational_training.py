#!/usr/bin/env python3
"""Command-line interface for foundational model training."""

import argparse
import json
from .train_foundational import train_foundational_model
from .training_config import TrainingConfig


def main():
    """Train foundational model from command line."""
    parser = argparse.ArgumentParser(
        description="Train foundational arithmetic LLM model"
    )
    
    # Required arguments
    parser.add_argument(
        "--corpus-path",
        type=str,
        required=True,
        help="Path to training corpus file"
    )
    
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path to trained tokenizer directory"
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
        default=1e-4,
        help="Learning rate (default: 1e-4)"
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
        default=10,
        help="Number of training epochs (default: 10)"
    )
    
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Number of warmup steps (default: 1000)"
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
        default=1000,
        help="Save checkpoint every N steps (default: 1000)"
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
        help="Path to model configuration JSON file"
    )
    
    parser.add_argument(
        "--d-model",
        type=int,
        default=256,
        help="Model embedding dimension (default: 256)"
    )
    
    parser.add_argument(
        "--nhead",
        type=int,
        default=8,
        help="Number of attention heads (default: 8)"
    )
    
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of transformer layers (default: 6)"
    )
    
    parser.add_argument(
        "--dim-feedforward",
        type=int,
        default=1024,
        help="Feedforward dimension (default: 1024)"
    )
    
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (default: 0.1)"
    )
    
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
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
    
    # Load or create model configuration
    if args.model_config:
        print(f"Loading model configuration from: {args.model_config}")
        with open(args.model_config, 'r') as f:
            model_config = json.load(f)
    else:
        model_config = {
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_layers': args.num_layers,
            'dim_feedforward': args.dim_feedforward,
            'dropout': args.dropout,
            'max_seq_length': args.max_seq_length
        }
    
    # Display configuration
    print("\n" + "=" * 60)
    print("FOUNDATIONAL MODEL TRAINING")
    print("=" * 60)
    print(f"\nCorpus: {args.corpus_path}")
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"Output directory: {args.output_dir}")
    print("\nTraining Configuration:")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Warmup steps: {config.warmup_steps}")
    print(f"  Gradient clip: {config.gradient_clip}")
    print(f"  Save every: {config.save_every} steps")
    print(f"  Device: {config.device}")
    print("\nModel Configuration:")
    print(f"  d_model: {model_config['d_model']}")
    print(f"  nhead: {model_config['nhead']}")
    print(f"  num_layers: {model_config['num_layers']}")
    print(f"  dim_feedforward: {model_config['dim_feedforward']}")
    print(f"  dropout: {model_config['dropout']}")
    print(f"  max_seq_length: {model_config['max_seq_length']}")
    print("=" * 60 + "\n")
    
    # Train model
    try:
        final_checkpoint = train_foundational_model(
            corpus_path=args.corpus_path,
            tokenizer_path=args.tokenizer_path,
            output_dir=args.output_dir,
            config=config,
            model_config=model_config
        )
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Final checkpoint: {final_checkpoint}")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("TRAINING FAILED!")
        print("=" * 60)
        print(f"Error: {str(e)}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()

