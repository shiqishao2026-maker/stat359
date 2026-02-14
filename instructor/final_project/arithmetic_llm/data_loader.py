"""Data loading utilities for arithmetic LLM training."""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from .arithmetic_tokenizer import (
    ArithmeticBPETokenizer
)
from .evaluator import eval_expression


class ArithmeticDataset(Dataset):
    """Dataset for arithmetic expressions and evaluations.
    
    Foundational mode loads plain text (one instance per line).
    Instruction mode loads JSONL entries for training.
    """
     
    def __init__(
        self,
        corpus_path: str,
        tokenizer: ArithmeticBPETokenizer,
        max_length: int = 512,
        mode: str = "foundational"
    ):
        """Initialize dataset.
        
        Args:
            corpus_path: Path to corpus file (text for foundational, JSONL for instruction)
            tokenizer: Trained tokenizer instance
            max_length: Maximum sequence length
            mode: Training mode - "foundational" or "instruction"
        """
        self.corpus_path = corpus_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.entries = []
        self.prompt_lengths = []  # Store prompt lengths for instruction mode
        
        # Load corpus
        self._load_corpus()

    def get_instruction_pairs(self, validate_expressions: bool = False) -> List[dict]:
        """Return instruction prompts and ground-truth answers.

        Each item has keys: 'prompt' and 'ground_truth'. This uses the
        instruction JSONL entries loaded in the dataset.
        """
        pairs: List[dict] = []
        if self.mode != "instruction":
            return pairs

        with open(self.corpus_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                problem = entry.get("problem")
                answer = entry.get("answer")
                if problem is None or answer is None:
                    continue
                if answer == "ERROR":
                    continue

                prompt = problem + " <think>"
                try:
                    ground_truth = int(answer)
                except (TypeError, ValueError):
                    continue

                if validate_expressions:
                    expression = entry.get("expression")
                    if not expression and isinstance(problem, str):
                        prefix = "Evaluate:"
                        if problem.startswith(prefix):
                            expression = problem[len(prefix):].strip()
                    if not expression:
                        continue
                    expected = eval_expression(expression).get("answer")
                    if expected == "ERROR":
                        continue
                    try:
                        expected_value = int(expected)
                    except (TypeError, ValueError):
                        continue
                    if expected_value != ground_truth:
                        continue

                pairs.append({"prompt": prompt, "ground_truth": ground_truth})

        return pairs
    
    def _load_corpus(self) -> None:
        """Load corpus.
        
        Foundational mode: treat each non-empty line as a standalone text instance.
        Instruction mode: each line is a JSON object with keys:
        - expression: The arithmetic expression
        - problem: "Evaluate: [expression]"
        - solution: The full solution with <think> tags
        - answer: The final numeric answer or "ERROR"
        """
        with open(self.corpus_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if self.mode == "foundational":
                    # Foundational: raw text per line
                    self.entries.append(line)
                    self.prompt_lengths.append(0)
                    continue

                try:
                    entry = json.loads(line)

                    # Instruction mode formatting
                    # prompt is problem + "<think>", target is solution
                    prompt = entry['problem'] + ' <think>'
                    target = entry['solution']

                    # Strip leading '<think>' from target if present
                    target = target.strip()
                    if target.startswith('<think>'):
                        # Keep the <think> tag for consistency
                        pass

                    # Concatenate prompt and target for training
                    text = prompt + ' ' + target

                    # Calculate prompt length in tokens (without BOS/EOS)
                    prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
                    prompt_length = len(prompt_tokens)

                    self.entries.append(text)
                    self.prompt_lengths.append(prompt_length)

                except json.JSONDecodeError:
                    # Skip malformed JSON lines in instruction mode
                    continue
    
    def __len__(self) -> int:
        """Return number of entries in dataset."""
        return len(self.entries)
    
    def __getitem__(self, idx: int) -> dict:
        """Get tokenized entry at index.
        
        Args:
            idx: Index of entry
            
        Returns:
            Dictionary with 'input_ids', 'attention_mask', 'length', and 'prompt_length'
        """
        entry = self.entries[idx]
        prompt_length = self.prompt_lengths[idx]
        
        # Tokenize entry (BOS and EOS will be added automatically)
        token_ids = self.tokenizer.encode(entry, add_special_tokens=True)
        
        # Adjust prompt_length to account for BOS token
        # BOS is at position 0, so all positions shift by 1
        if prompt_length > 0:
            prompt_length += 1  # Account for BOS token
        
        # Truncate if necessary
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
            # Adjust prompt_length if truncated
            if prompt_length > self.max_length:
                prompt_length = self.max_length
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(token_ids)
        
        return {
            'input_ids': token_ids,
            'attention_mask': attention_mask,
            'length': len(token_ids),
            'prompt_length': prompt_length
        }


def collate_fn(
    batch: List[dict],
    pad_token_id: int = 0,
    mode: str = "foundational"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for batching with padding.
    
    Args:
        batch: List of dictionaries from dataset
        pad_token_id: Token ID to use for padding
        mode: Training mode - "foundational" or "instruction"
        
    Returns:
        Tuple of (input_ids, attention_mask, labels) tensors
        For instruction mode, labels have -100 for prompt tokens (ignored in loss)
    """
    # Find maximum length in batch
    max_length = max(item['length'] for item in batch)
    
    # Pad sequences
    input_ids = []
    attention_masks = []
    labels = []
    
    for item in batch:
        # Pad input_ids
        padded_ids = item['input_ids'] + [pad_token_id] * (
            max_length - item['length']
        )
        input_ids.append(padded_ids)
        
        # Pad attention_mask
        padded_mask = item['attention_mask'] + [0] * (
            max_length - item['length']
        )
        attention_masks.append(padded_mask)
        
        # Create labels for loss computation
        if mode == "instruction":
            # For instruction mode: mask prompt tokens with -100
            prompt_length = item['prompt_length']
            # Labels are shifted by 1 (predict next token)
            # Mask prompt tokens (including the shift)
            label_ids = [-100] * prompt_length + item['input_ids'][prompt_length:]
            # Pad labels
            label_ids = label_ids + [-100] * (max_length - item['length'])
        else:
            # For foundational mode: use all tokens
            label_ids = item['input_ids'] + [-100] * (max_length - item['length'])
        
        labels.append(label_ids)
    
    # Convert to tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return input_ids, attention_masks, labels


def create_dataloaders(
    corpus_path: str,
    tokenizer: ArithmeticBPETokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    train_split: float = 0.9,
    shuffle: bool = True,
    num_workers: int = 2,  # Changed from 0 to 2 for parallel loading
    mode: str = "foundational"
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.
    
    Args:
        corpus_path: Path to corpus file (text for foundational, JSONL for instruction)
        tokenizer: Trained tokenizer instance
        batch_size: Batch size for dataloaders
        max_length: Maximum sequence length
        train_split: Fraction of data to use for training
        shuffle: Whether to shuffle training data
        num_workers: Number of worker processes for data loading
        mode: Training mode - "foundational" or "instruction"
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create full dataset
    full_dataset = ArithmeticDataset(
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        max_length=max_length,
        mode=mode
    )
    
    # Split into train and validation
    # Ensure at least 1 sample in each split if dataset is large enough
    dataset_size = len(full_dataset)
    
    if dataset_size == 0:
        raise ValueError("Dataset is empty")
    
    if dataset_size == 1:
        # If only 1 sample, use it for both train and val
        train_size = 1
        val_size = 1
        train_dataset = full_dataset
        val_dataset = full_dataset
    else:
        # Ensure at least 1 sample in validation set
        train_size = max(1, int(dataset_size * train_split))
        val_size = max(1, dataset_size - train_size)
        
        # Adjust train_size if needed to ensure both sets have at least 1 sample
        if train_size + val_size > dataset_size:
            train_size = dataset_size - 1
            val_size = 1
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size]
        )
    
    # Get pad token ID
    pad_token_id = tokenizer.token2id.get('<pad>', 0)
    
    # Create collate function with pad token ID and mode
    def collate_with_pad(batch):
        return collate_fn(batch, pad_token_id=pad_token_id, mode=mode)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_with_pad
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_with_pad
    )
    
    return train_dataloader, val_dataloader


