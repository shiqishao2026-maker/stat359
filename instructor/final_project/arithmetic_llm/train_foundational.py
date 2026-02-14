"""Foundational model training script for arithmetic LLM.

This module implements the training pipeline for the foundational transformer model
on arithmetic expressions and evaluations.
"""

import os
import json
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from typing import Optional, Dict, Tuple

from .transformer_model import ArithmeticTransformer
from .arithmetic_tokenizer import ArithmeticBPETokenizer
from .data_loader import create_dataloaders
from .training_config import TrainingConfig


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create learning rate scheduler with linear warmup and decay.
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        last_epoch: Last epoch number for resuming training
        
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / 
            float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def save_checkpoint(
    model: ArithmeticTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    step: int,
    loss: float,
    config: TrainingConfig,
    tokenizer_vocab_size: int,
    output_dir: str,
    is_final: bool = False
) -> str:
    """Save model checkpoint with metadata.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        scheduler: Scheduler state to save
        epoch: Current epoch number
        step: Current training step
        loss: Current training loss
        config: Training configuration
        tokenizer_vocab_size: Size of tokenizer vocabulary
        output_dir: Directory to save checkpoint
        is_final: Whether this is the final checkpoint
        
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'config': config.to_dict(),
        'model_config': {
            'vocab_size': model.vocab_size,
            'd_model': model.d_model,
            'nhead': model.nhead,
            'num_layers': model.num_layers,
            'dim_feedforward': model.dim_feedforward,
            'dropout': model.dropout,
            'max_seq_length': model.max_seq_length,
        },
        'tokenizer_vocab_size': tokenizer_vocab_size,
    }
    
    if is_final:
        checkpoint_path = os.path.join(output_dir, 'final_model.pt')
    else:
        checkpoint_path = os.path.join(output_dir, f'checkpoint_step_{step}.pt')
    
    torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: ArithmeticTransformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
) -> Dict:
    """Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        
    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'step': checkpoint.get('step', 0),
        'loss': checkpoint.get('loss', 0.0),
        'config': checkpoint.get('config', {}),
        'tokenizer_vocab_size': checkpoint.get('tokenizer_vocab_size', 0),
    }


def train_epoch(
    model: ArithmeticTransformer,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: TrainingConfig,
    epoch: int,
    global_step: int,
    output_dir: str,
    tokenizer_vocab_size: int
) -> Tuple[float, int]:
    """Train model for one epoch.
    
    Args:
        model: Model to train
        train_dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Training configuration
        epoch: Current epoch number
        global_step: Current global step
        output_dir: Directory to save checkpoints
        tokenizer_vocab_size: Size of tokenizer vocabulary
        
    Returns:
        Tuple of (average_loss, final_global_step)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (input_ids, attention_mask, labels) in enumerate(progress_bar):
        # Move to device
        input_ids = input_ids.to(config.device)
        attention_mask = attention_mask.to(config.device)
        labels = labels.to(config.device)
        
        # Prepare inputs and targets for next-token prediction
        # Input: all tokens except last, Target: all tokens except first (shifted)
        inputs = input_ids[:, :-1]
        targets = labels[:, 1:]  # Use labels which have prompt masking for instruction mode
        input_attention_mask = attention_mask[:, :-1]
        
        # Forward pass
        logits = model(inputs, input_attention_mask)
        
        # Compute loss
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100  # Ignore masked tokens (padding and prompt in instruction mode)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        
        # Update statistics (accumulate on GPU, sync less frequently)
        total_loss += loss.detach()  # Keep on GPU
        num_batches += 1
        global_step += 1
        
        # Update progress bar (only sync every 10 batches to avoid slowdown)
        if batch_idx % 10 == 0:
            progress_bar.set_postfix({
                'loss': loss.item(),
                'avg_loss': (total_loss / num_batches).item(),
                'lr': scheduler.get_last_lr()[0]
            })
        
        # Save checkpoint at intervals
        if global_step % config.save_every == 0:
            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=global_step,
                loss=loss.item(),
                config=config,
                tokenizer_vocab_size=tokenizer_vocab_size,
                output_dir=output_dir,
                is_final=False
            )
            print(f"\nCheckpoint saved at step {global_step}: {checkpoint_path}")
    
    avg_loss = (total_loss / num_batches).item() if num_batches > 0 else 0.0
    return avg_loss, global_step


def evaluate(
    model: ArithmeticTransformer,
    val_dataloader: torch.utils.data.DataLoader,
    config: TrainingConfig
) -> float:
    """Evaluate model on validation set.
    
    Args:
        model: Model to evaluate
        val_dataloader: Validation data loader
        config: Training configuration
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_dataloader:
            # Move to device
            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            labels = labels.to(config.device)
            
            # Prepare inputs and targets
            inputs = input_ids[:, :-1]
            targets = labels[:, 1:]
            input_attention_mask = attention_mask[:, :-1]
            
            # Forward pass
            logits = model(inputs, input_attention_mask)
            
            # Compute loss
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100
            )
            
            total_loss += loss.detach()  # Keep on GPU
            num_batches += 1
    
    avg_loss = (total_loss / num_batches).item() if num_batches > 0 else 0.0
    return avg_loss


def train_foundational_model(
    corpus_path: str,
    tokenizer_path: str,
    output_dir: str,
    config: TrainingConfig,
    model_config: Optional[Dict] = None
) -> str:
    """Train foundational model on arithmetic corpus.
    
    Args:
        corpus_path: Path to training corpus
        tokenizer_path: Path to trained tokenizer
        output_dir: Directory to save checkpoints and logs
        config: Training configuration
        model_config: Optional model architecture configuration
        
    Returns:
        Path to final model checkpoint
    """
    # Validate configuration
    config.validate()
    
    # Create unique output directory with timestamp (including microseconds for uniqueness)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = os.path.join(output_dir, f"foundational_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Training output directory: {output_dir}")
    print(f"Configuration: {config.to_dict()}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = ArithmeticBPETokenizer()
    tokenizer.load(tokenizer_path)
    vocab_size = len(tokenizer.token2id)
    print(f"Tokenizer vocabulary size: {vocab_size}")
    
    # Initialize model config
    print("Initializing model configuration...")
    if model_config is None:
        model_config = {
            'vocab_size': vocab_size,
            'd_model': 256,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 1024,
            'dropout': 0.1,
            'max_seq_length': 512
        }
    else:
        model_config['vocab_size'] = vocab_size

    max_seq_length = model_config.get('max_seq_length', 512)

    # Create dataloaders
    print("Creating dataloaders...")
    train_dataloader, val_dataloader = create_dataloaders(
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=max_seq_length,
        train_split=0.9,
        shuffle=True,
        num_workers=0,
        mode="foundational"
    )
    print(f"Training batches: {len(train_dataloader)}")
    print(f"Validation batches: {len(val_dataloader)}")
    
    # Initialize model
    print("Initializing model...")
    
    model = ArithmeticTransformer(**model_config)
    model = model.to(config.device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
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
    
    # Training loop
    print("\nStarting training...")
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
        
        # Save best model
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
            # Rename to best_model.pt
            best_model_path = os.path.join(output_dir, 'best_model.pt')
            os.rename(best_checkpoint_path, best_model_path)
            print(f"  New best model saved: {best_model_path}")
    
    # Save final model
    print("\nSaving final model...")
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
    print(f"Final model saved: {final_checkpoint_path}")
    
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
        'tokenizer_vocab_size': vocab_size
    }
    summary_path = os.path.join(output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary saved: {summary_path}")
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    return final_checkpoint_path


