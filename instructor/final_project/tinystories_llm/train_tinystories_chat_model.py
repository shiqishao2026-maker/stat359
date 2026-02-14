import os
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from datasets import load_dataset
from bpe_tokenizer import BPETokenizer
from transformer_model import TinyStoriesConfig, TinyStoriesForCausalLM
from tqdm import tqdm
import time
import json
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# === Parse Arguments ===
def parse_args():
    parser = argparse.ArgumentParser(description="Train a TinyStories chat model with instruction tuning")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="bochen0909/tinystories-conversations", help="HuggingFace dataset name")
    parser.add_argument("--tokenizer_path", type=str, default="bpe_tokenizer_tinystories.pkl", help="Path to BPE tokenizer")
    parser.add_argument("--max_seq_len", type=int, default=256, help="Maximum sequence length")
    
    # Model architecture arguments
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size of the model")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--intermediate_size", type=int, default=1024, help="Size of the intermediate (feed-forward) layer")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--window_size", type=int, default=256, help="Attention window size for local attention")
    
    # Training arguments (UPDATED DEFAULTS FOR INSTRUCTION TUNING)
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (reduced for instruction tuning)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (lower for instruction tuning)")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps for learning rate scheduler (lower for instruction tuning)")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay (lower for instruction tuning)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for gradient clipping")
    
    # Instruction tuning arguments
    parser.add_argument("--pretrained_model_path",  required=True, type=str, default=None, help="Path to pretrained model checkpoint")
    parser.add_argument("--user_token", type=str, default="<user>", help="Token to represent user messages")
    parser.add_argument("--assistant_token", type=str, default="<assistant>", help="Token to represent assistant messages")
    
    # Misc arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="tinystories_chat_model", help="Output directory for model checkpoints")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=3000, help="Evaluate every X steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X steps")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Maximum number of training samples (for debugging)")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Maximum number of evaluation samples (for debugging)")
    parser.add_argument("--pilot_run", action="store_true", help="If set, use a small number of samples for a quick pilot run (overrides max_train_samples and max_eval_samples)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device to use: auto, cpu, cuda, or mps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision (AMP) training (float16 on CUDA)")
    
    return parser.parse_args()

# === Set Device ===
def get_device(device_preference):
    if device_preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif device_preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            raise ValueError("CUDA requested but not available.")
    elif device_preference == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        else:
            raise ValueError("MPS requested but not available.")
    else:
        return torch.device("cpu")

# === Set Seed ===
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# === Load Tokenizer ===
def load_tokenizer(tokenizer_path):
    return BPETokenizer.load(tokenizer_path)

# === Dataset ===
class TinyStoriesConversationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512, split="train", max_samples=None, 
                 user_token="<user>", assistant_token="<assistant>"):
        self.dataset = dataset[split]
        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.user_token = user_token
        self.assistant_token = assistant_token
        
        # Ensure special tokens are in tokenizer
        if user_token not in tokenizer.token2id:
            raise ValueError(f"User token {user_token} not found in tokenizer vocabulary")
        if assistant_token not in tokenizer.token2id:
            raise ValueError(f"Assistant token {assistant_token} not found in tokenizer vocabulary")
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get conversation data
        conversation = self.dataset[idx]["conversation"]
        
        # Format conversation as alternating user/assistant turns
        # Ignore the "name" field as instructed
        formatted_text = ""
        for i, turn in enumerate(conversation):
            # Alternate between user and assistant roles
            if i % 2 == 0:  # Even indices are user turns
                formatted_text += f"{self.user_token} {turn['text']} "
            else:  # Odd indices are assistant turns
                formatted_text += f"{self.assistant_token} {turn['text']} "
        
        # Tokenize the formatted conversation
        tokens = self.tokenizer.encode(formatted_text, add_special_tokens=False)
        
        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            pad_len = self.max_length - len(tokens)
            tokens += [self.tokenizer.token2id.get('<pad>', 0)] * pad_len
            
        return torch.tensor(tokens, dtype=torch.long)

# === Learning Rate Scheduler ===
class WarmupLinearScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr_scale = float(self.current_step) / float(max(1, self.warmup_steps))
        else:
            progress = float(self.current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            lr_scale = max(0.0, 1.0 - progress)
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * lr_scale

# === Training and Evaluation ===
def train_and_evaluate(args):
    # Set device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=args.output_dir)
    
    # Save args for reproducibility
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer_path)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset)
    
    # Create train and validation datasets
    train_dataset = TinyStoriesConversationDataset(
        dataset, tokenizer, max_length=args.max_seq_len, 
        split="train", max_samples=args.max_train_samples,
        user_token=args.user_token, assistant_token=args.assistant_token
    )
    val_dataset = TinyStoriesConversationDataset(
        dataset, tokenizer, max_length=args.max_seq_len, 
        split="valid", max_samples=args.max_eval_samples,
        user_token=args.user_token, assistant_token=args.assistant_token
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Create model configuration
    config = TinyStoriesConfig(
        vocab_size=len(tokenizer.token2id),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        max_position_embeddings=args.max_seq_len,
        window_size=args.window_size,
    )
    
    # Create model
    model = TinyStoriesForCausalLM(config).to(device)
    
    # Load pretrained model if specified
    if args.pretrained_model_path is not None:
        print(f"Loading pretrained model from {args.pretrained_model_path}")
        model.load_state_dict(torch.load(args.pretrained_model_path, map_location=device))
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model architecture: {args.num_layers} layers, {args.hidden_size} hidden size, {args.num_heads} attention heads")
    print(f"Total parameters: {total_params:,}")
    try:
        dummy_input = torch.randint(0, config.vocab_size, (1, args.max_seq_len-1), dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=dummy_input)
        print(f"Dummy input shape: {dummy_input.shape}, Output logits shape: {out['logits'].shape}")
    except Exception as e:
        print(f"Dummy forward pass failed: {e}")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Set initial learning rate for scheduler
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.lr
    
    # Create learning rate scheduler
    total_steps = len(train_dataloader) * args.epochs // args.gradient_accumulation_steps
    scheduler = WarmupLinearScheduler(optimizer, args.warmup_steps, total_steps)
    
    # Create loss function
    pad_token_id = tokenizer.token2id.get('<pad>', 0)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    train_losses = []
    if args.resume_from_checkpoint is not None and os.path.isfile(args.resume_from_checkpoint):
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and 'current_step' in checkpoint['scheduler_state_dict']:
            scheduler.current_step = checkpoint['scheduler_state_dict']['current_step']
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'global_step' in checkpoint:
            global_step = checkpoint['global_step']
        print(f"Resumed at epoch {start_epoch}, global step {global_step}")
    
    # AMP scaler (only if using CUDA and AMP is enabled)
    use_amp = args.amp and device.type == "cuda"
    if use_amp:
        from torch.amp import autocast, GradScaler
        scaler = GradScaler(device="cuda")
    else:
        autocast = None
        scaler = None
    # Set AMP flag on model for evaluation
    if use_amp:
        model.use_amp = True
    else:
        model.use_amp = False
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Training epoch {epoch+1}")
        for step, batch in enumerate(progress_bar):
            batch = batch.to(device)
            
            # Get inputs and targets (shift right for causal language modeling)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            # Forward pass with AMP if enabled
            if use_amp:
                with autocast(device_type="cuda"):
                    outputs = model(input_ids=inputs)
                    logits = outputs["logits"]
                    loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    loss = loss / args.gradient_accumulation_steps
            else:
                outputs = model(input_ids=inputs)
                logits = outputs["logits"]
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                loss = loss / args.gradient_accumulation_steps

            # Backward pass
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights if gradient accumulation steps reached
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Clip gradients
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # Update weights
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Update global step
                global_step += 1
                
                # Log training loss
                if global_step % args.logging_steps == 0:
                    train_losses.append(loss.item() * args.gradient_accumulation_steps)
                    avg_loss = sum(train_losses[-100:]) / min(len(train_losses), 100)
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                    # Log to TensorBoard
                    writer.add_scalar('Loss/train', avg_loss, global_step)
                    writer.add_scalar('Perplexity/train', np.exp(avg_loss), global_step)
                
                # Evaluate
                if global_step % args.eval_steps == 0:
                    val_loss = evaluate(model, val_dataloader, criterion, device)
                    val_ppl = np.exp(val_loss)
                    print(f"Step {global_step}: Validation loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}")
                    writer.add_scalar('Loss/val', val_loss, global_step)
                    writer.add_scalar('Perplexity/val', val_ppl, global_step)
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        model_path = os.path.join(args.output_dir, "best_model.pth")
                        torch.save(model.state_dict(), model_path)
                        print(f"New best model saved to {model_path}")
                    # Back to training mode
                    model.train()
                
                # Save checkpoint
                if global_step % args.save_steps == 0:
                    checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pth")
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': {
                            'current_step': scheduler.current_step,
                        },
                        'loss': loss.item(),
                    }, checkpoint_path)
                    print(f"Checkpoint saved to {checkpoint_path}")
            
            # Update epoch loss
            epoch_loss += loss.item() * args.gradient_accumulation_steps
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        train_ppl = np.exp(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{args.epochs}, Average Train Loss: {avg_epoch_loss:.4f}, Train Perplexity: {train_ppl:.2f}")
        writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch+1)
        writer.add_scalar('Perplexity/train_epoch', train_ppl, epoch+1)
        val_loss = evaluate(model, val_dataloader, criterion, device)
        val_ppl = np.exp(val_loss)
        print(f"Epoch {epoch+1}/{args.epochs}, Validation Loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}")
        writer.add_scalar('Loss/val_epoch', val_loss, epoch+1)
        writer.add_scalar('Perplexity/val_epoch', val_ppl, epoch+1)
        model_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    writer.close()
    
    return model, device

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            if hasattr(model, 'use_amp') and model.use_amp:
                from torch.amp import autocast
                with autocast(device_type="cuda"):
                    outputs = model(input_ids=inputs)
                    logits = outputs["logits"]
                    loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            else:
                outputs = model(input_ids=inputs)
                logits = outputs["logits"]
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def generate_chat_response(model, tokenizer, prompt, device, max_length=100, temperature=0.7, top_p=0.9):
    """Generate a chat response for a given prompt."""
    model.eval()
    
    # Format prompt with user token
    user_token = tokenizer.token2id.get('<user>', None)
    assistant_token = tokenizer.token2id.get('<assistant>', None)
    
    if user_token is None or assistant_token is None:
        raise ValueError("User or assistant token not found in tokenizer vocabulary")
    
    # Encode prompt with user token
    formatted_prompt = f"<user> {prompt} <assistant>"
    input_ids = torch.tensor([tokenizer.encode(formatted_prompt, add_special_tokens=True)]).to(device)
    
    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
        )
    
    # Decode output
    output_text = tokenizer.decode(output_ids[0].tolist())
    
    # Extract assistant's response
    try:
        assistant_response = output_text.split('<assistant>')[-1].strip()
        # Remove any trailing user tokens
        if '<user>' in assistant_response:
            assistant_response = assistant_response.split('<user>')[0].strip()
    except:
        assistant_response = output_text
    
    return assistant_response
 
if __name__ == '__main__':
    args = parse_args()
    # Handle pilot run option
    if getattr(args, 'pilot_run', False):
        args.max_train_samples = 1000
        args.max_eval_samples = 100
        print("[Pilot Run] Using 1000 samples for training and 100 for evaluation.")
    
    start_time = time.time()
    model, device = train_and_evaluate(args)
    end_time = time.time()
    
    print(f"\nTraining completed in {(end_time - start_time) / 60:.2f} minutes")
    
