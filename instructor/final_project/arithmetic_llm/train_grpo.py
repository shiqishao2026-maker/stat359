"""Training entry point for GRPO."""

from typing import Any, Dict, Iterator, List, Optional, Tuple
import os

import math
from datetime import datetime

from .arithmetic_tokenizer import ArithmeticBPETokenizer
from .data_loader import ArithmeticDataset
from .evaluator import eval_expression
from .generator import ExpressionGenerator
from .grpo_config import GRPOConfig
from .grpo_trainer import GRPOTrainer


def _batch_iter(items: List[dict], batch_size: int) -> Iterator[Tuple[List[str], List[int]]]:
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        prompts = [entry["prompt"] for entry in batch]
        ground_truth = [entry["ground_truth"] for entry in batch]
        yield prompts, ground_truth


def _load_instruction_pairs(
    instruction_corpus_path: str,
    tokenizer: ArithmeticBPETokenizer,
    validate_expressions: bool
) -> List[dict]:
    dataset = ArithmeticDataset(
        corpus_path=instruction_corpus_path,
        tokenizer=tokenizer,
        mode="instruction"
    )
    return dataset.get_instruction_pairs(validate_expressions=validate_expressions)


def _generate_pairs(
    num_samples: int,
    max_depth: int,
    num_range: Tuple[int, int]
) -> List[dict]:
    generator = ExpressionGenerator(
        max_depth=max_depth,
        num_range=num_range,
        invalid_rate=0.0
    )
    pairs = []
    for _ in range(num_samples):
        expression = generator.generate()
        result = eval_expression(expression)
        if result["answer"] == "ERROR":
            continue
        pairs.append({
            "prompt": result["problem"] + " <think>",
            "ground_truth": int(result["answer"])
        })
    return pairs


def train_grpo_model(
    instruction_corpus_path: Optional[str],
    tokenizer_path: str,
    sft_checkpoint_path: str,
    output_dir: str,
    config: GRPOConfig,
    data_mode: str = "instruction",
    num_samples: int = 1000,
    max_depth: int = 5,
    num_range: Tuple[int, int] = (1, 20),
    filter_invalid_instruction: bool = True,
    candidate_sub_batch_size: Optional[int] = None
) -> Dict[str, Any]:
    """Train GRPO model using instruction corpus or generated data."""
    tokenizer = ArithmeticBPETokenizer()
    tokenizer.load(tokenizer_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = os.path.join(output_dir, f"grpo_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"GRPO output directory: {output_dir}")

    trainer = GRPOTrainer(
        config=config,
        sft_checkpoint_path=sft_checkpoint_path,
        tokenizer=tokenizer,
        candidate_sub_batch_size=candidate_sub_batch_size,
    )

    if data_mode == "instruction":
        if instruction_corpus_path is None:
            raise ValueError("instruction_corpus_path required for instruction mode")
        pairs = _load_instruction_pairs(
            instruction_corpus_path,
            tokenizer,
            filter_invalid_instruction
        )
    elif data_mode == "generated":
        pairs = _generate_pairs(num_samples, max_depth, num_range)
    else:
        raise ValueError("data_mode must be 'instruction' or 'generated'")

    if not pairs:
        raise ValueError("No training pairs available")

    total_steps = math.ceil(len(pairs) / config.batch_size) * config.num_epochs
    trainer.reset_optimizer_and_scheduler(total_steps=total_steps)

    train_dataloader = list(_batch_iter(pairs, config.batch_size))
    return trainer.train(train_dataloader, output_dir=output_dir)


