#!/usr/bin/env python3
"""Command-line utility for merging LoRA adapters into a base model."""

import argparse

from .lora_utils import merge_lora_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base checkpoint"
    )
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        required=True,
        help="Path to base model checkpoint"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to LoRA adapter file"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save merged model"
    )

    args = parser.parse_args()

    merge_lora_checkpoint(
        base_checkpoint_path=args.base_checkpoint,
        adapter_path=args.adapter_path,
        output_path=args.output_path
    )

    print(f"Merged model saved to: {args.output_path}")


if __name__ == "__main__":
    main()


