"""CLI for GRPO training."""

import argparse
import os

from .grpo_config import GRPOConfig
from .train_grpo import train_grpo_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GRPO training")
    parser.add_argument(
        "--instruction-corpus",
        type=str,
        default=None,
        help="Path to instruction corpus JSONL (required for instruction mode)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to tokenizer directory"
    )
    parser.add_argument(
        "--sft-checkpoint",
        type=str,
        required=True,
        help="Path to SFT checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save checkpoints and logs"
    )
    parser.add_argument(
        "--data-mode",
        type=str,
        choices=["instruction", "generated"],
        default="instruction",
        help="Training data mode"
    )
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--kl-penalty-coef", type=float, default=0.05)
    parser.add_argument("--max-gen-length", type=int, default=512)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--candidate-sub-batch-size",
        type=int,
        default=None,
        help="Optional sub-batch size for candidate processing"
    )
    parser.add_argument(
        "--filter-invalid-instruction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter instruction entries with invalid/mismatched expressions"
    )
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--num-range-min", type=int, default=1)
    parser.add_argument("--num-range-max", type=int, default=20)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.data_mode == "instruction" and not args.instruction_corpus:
        raise ValueError("--instruction-corpus is required for instruction mode")
    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(args.tokenizer)
    if not os.path.exists(args.sft_checkpoint):
        raise FileNotFoundError(args.sft_checkpoint)
    if args.num_range_min > args.num_range_max:
        raise ValueError("num-range-min must be <= num-range-max")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args)

    config = GRPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        gradient_clip=args.gradient_clip,
        save_every=args.save_every,
        eval_every=args.eval_every,
        num_candidates=args.num_candidates,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        kl_penalty_coef=args.kl_penalty_coef,
        max_gen_length=args.max_gen_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_every=args.log_every,
    )
    config.validate()

    train_grpo_model(
        instruction_corpus_path=args.instruction_corpus,
        tokenizer_path=args.tokenizer,
        sft_checkpoint_path=args.sft_checkpoint,
        output_dir=args.output_dir,
        config=config,
        data_mode=args.data_mode,
        num_samples=args.num_samples,
        max_depth=args.max_depth,
        num_range=(args.num_range_min, args.num_range_max),
        filter_invalid_instruction=args.filter_invalid_instruction,
        candidate_sub_batch_size=args.candidate_sub_batch_size,
    )


if __name__ == "__main__":
    main()

