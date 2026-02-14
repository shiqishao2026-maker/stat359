"""LoRA instruction fine-tuning script for arithmetic LLM."""

import os
import json
import torch
from datetime import datetime
from typing import Optional, Dict

from .transformer_model import ArithmeticTransformer
from .arithmetic_tokenizer import ArithmeticBPETokenizer
from .data_loader import create_dataloaders
from .training_config import TrainingConfig
from .lora_config import LoRAConfig
from .lora_utils import get_parameter_stats
from .train_foundational import (
    get_linear_schedule_with_warmup,
    save_checkpoint,
    load_checkpoint,
    train_epoch,
    evaluate,
)


def freeze_non_lora_parameters(model: ArithmeticTransformer) -> None:
    """Freeze all parameters except LoRA adapters."""
    for param in model.parameters():
        param.requires_grad = False

    for param in model.get_lora_parameters():
        param.requires_grad = True


def create_lora_optimizer(
    model: ArithmeticTransformer,
    config: TrainingConfig
) -> torch.optim.Optimizer:
    """Create optimizer for LoRA parameters only."""
    lora_params = list(model.get_lora_parameters())
    if not lora_params:
        raise ValueError("No LoRA parameters found for optimization")

    return torch.optim.AdamW(
        lora_params,
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )



def train_instruction_model_lora(
    instruction_corpus_path: str,
    tokenizer_path: str,
    foundational_checkpoint: str,
    output_dir: str,
    config: TrainingConfig,
    lora_config: Optional[LoRAConfig] = None,
    model_config: Optional[Dict] = None,
    save_merged_model: bool = False
) -> str:
    """Fine-tune model with LoRA on instruction-formatted data.

    Returns:
        Path to saved LoRA adapter file
    """
    # Validate configuration
    config.validate()

    if lora_config is None:
        lora_config = config.lora_config

    if lora_config is None:
        raise ValueError("LoRA configuration is required for LoRA training")

    lora_config.validate()
    config.lora_config = lora_config

    # Create unique output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = os.path.join(output_dir, f"instruction_lora_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"LoRA fine-tuning output directory: {output_dir}")
    print(f"Training configuration: {config.to_dict()}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = ArithmeticBPETokenizer()
    tokenizer.load(tokenizer_path)
    vocab_size = len(tokenizer.token2id)
    print(f"Tokenizer vocabulary size: {vocab_size}")

    # Initialize model architecture
    print("Initializing model architecture...")
    checkpoint_data = torch.load(foundational_checkpoint, map_location='cpu')
    if model_config is None:
        model_config = checkpoint_data.get('model_config', {
            'vocab_size': vocab_size,
            'd_model': 256,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 1024,
            'dropout': 0.1,
            'max_seq_length': 512
        })
    else:
        model_config['vocab_size'] = vocab_size

    checkpoint_vocab_size = checkpoint_data.get('tokenizer_vocab_size')
    if checkpoint_vocab_size is not None and checkpoint_vocab_size != vocab_size:
        raise ValueError(
            f"Tokenizer vocab size ({vocab_size}) does not match checkpoint "
            f"vocab size ({checkpoint_vocab_size})."
        )

    model_config['vocab_size'] = vocab_size
    max_seq_length = model_config.get('max_seq_length', 512)

    # Create dataloaders
    print("Creating dataloaders...")
    train_dataloader, val_dataloader = create_dataloaders(
        corpus_path=instruction_corpus_path,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=max_seq_length,
        train_split=0.9,
        shuffle=True,
        num_workers=0,
        mode="instruction"
    )
    print(f"Training batches: {len(train_dataloader)}")
    print(f"Validation batches: {len(val_dataloader)}")

    model = ArithmeticTransformer(**model_config)

    # Load foundational model weights
    print(f"Loading foundational model from: {foundational_checkpoint}")
    checkpoint_metadata = load_checkpoint(
        checkpoint_path=foundational_checkpoint,
        model=model
    )
    print(f"Loaded checkpoint from epoch {checkpoint_metadata['epoch']}, "
          f"step {checkpoint_metadata['step']}")

    # Inject LoRA and freeze parameters
    model.inject_lora(lora_config)
    freeze_non_lora_parameters(model)

    model = model.to(config.device)

    # Count parameters
    stats = get_parameter_stats(model)
    print(f"Total parameters: {stats['total']:,}")
    print(f"Trainable parameters: {stats['trainable']:,}")
    print(f"Frozen parameters: {stats['frozen']:,}")
    print(f"Trainable percentage: {stats['trainable_pct']:.2f}%")

    # Initialize optimizer
    optimizer = create_lora_optimizer(model, config)

    # Calculate total training steps
    total_steps = len(train_dataloader) * config.num_epochs

    # Initialize scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )

    # Save configuration
    config.to_json(os.path.join(output_dir, 'training_config.json'))
    with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
        json.dump(model_config, f, indent=2)

    # Save foundational checkpoint path for reference
    with open(os.path.join(output_dir, 'foundational_checkpoint.txt'), 'w') as f:
        f.write(foundational_checkpoint)

    # Training loop
    print("\nStarting LoRA instruction fine-tuning...")
    global_step = 0
    best_val_loss = float('inf')
    training_log = []

    for epoch in range(config.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"{'='*60}")

        # Train for one epoch
        train_loss, global_step = train_epoch(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            epoch=epoch + 1,
            global_step=global_step,
            output_dir=output_dir,
            tokenizer_vocab_size=vocab_size
        )

        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        val_loss = evaluate(model, val_dataloader, config)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Training Loss: {train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")

        # Log metrics
        training_log.append({
            'epoch': epoch + 1,
            'step': global_step,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': scheduler.get_last_lr()[0]
        })

        # Save best model checkpoint (LoRA weights included)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                step=global_step,
                loss=val_loss,
                config=config,
                tokenizer_vocab_size=vocab_size,
                output_dir=output_dir,
                is_final=False
            )
            best_model_path = os.path.join(output_dir, 'best_model.pt')
            os.rename(best_checkpoint_path, best_model_path)
            print(f"  New best model saved: {best_model_path}")

    # Save final model checkpoint (LoRA weights included)
    print("\nSaving final fine-tuned model checkpoint...")
    final_checkpoint_path = save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=config.num_epochs,
        step=global_step,
        loss=train_loss,
        config=config,
        tokenizer_vocab_size=vocab_size,
        output_dir=output_dir,
        is_final=True
    )
    print(f"Final model checkpoint saved: {final_checkpoint_path}")

    # Save LoRA adapters separately
    adapter_path = os.path.join(output_dir, 'lora_adapter.pt')
    model.save_lora_adapters(adapter_path, base_model_path=foundational_checkpoint)
    print(f"LoRA adapter saved: {adapter_path}")

    merged_model_path = None
    if save_merged_model:
        print("Merging LoRA weights and saving merged model...")
        model.merge_lora_weights()
        merged_model_path = os.path.join(output_dir, 'merged_model.pt')
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'config': config.to_dict(),
                'tokenizer_vocab_size': vocab_size,
                'merged': True,
                'foundational_checkpoint': foundational_checkpoint,
            },
            merged_model_path
        )
        print(f"Merged model saved: {merged_model_path}")

    # Save training log
    log_path = os.path.join(output_dir, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"Training log saved: {log_path}")

    # Save summary
    summary = {
        'total_epochs': config.num_epochs,
        'total_steps': global_step,
        'final_train_loss': train_loss,
        'final_val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'model_config': model_config,
        'training_config': config.to_dict(),
        'tokenizer_vocab_size': vocab_size,
        'foundational_checkpoint': foundational_checkpoint,
        'lora_adapter_path': adapter_path,
        'merged_model_path': merged_model_path
    }
    summary_path = os.path.join(output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary saved: {summary_path}")

    print("\n" + "="*60)
    print("LoRA instruction fine-tuning completed successfully!")
    print(f"Output directory: {output_dir}")
    print("="*60)

    return adapter_path


