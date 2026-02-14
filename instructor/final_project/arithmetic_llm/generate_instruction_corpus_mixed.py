#!/usr/bin/env python3
"""Generate mixed instruction corpus (valid + invalid) without subprocesses."""

import argparse
import os
import random
import tempfile
from typing import List, Tuple

from .corpus_generator import CorpusGenerator


def _generate_instruction_corpus(
    num_samples: int,
    max_depth: int,
    num_range: Tuple[int, int],
    invalid_rate: float,
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    generator = CorpusGenerator(
        num_samples=num_samples,
        max_depth=max_depth,
        num_range=num_range,
        invalid_rate=invalid_rate,
        output_path=output_path,
    )
    generator.generate_instruction_corpus(output_path)


def _read_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line for line in f if line.strip()]


def _write_lines(path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for line in lines:
            f.write(line if line.endswith("\n") else line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a mixed instruction corpus with valid and invalid samples."
    )
    parser.add_argument("--num-samples", type=int, default=20000)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--num-range", type=int, nargs=2, default=[1, 20])
    parser.add_argument("--invalid-rate", type=float, default=0.1)
    parser.add_argument(
        "--output-mixed",
        type=str,
        default="data/instruction_corpus.txt",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.num_samples <= 0:
        parser.error("num-samples must be positive")
    if args.max_depth <= 0:
        parser.error("max-depth must be positive")
    if args.num_range[0] >= args.num_range[1]:
        parser.error("num-range MIN must be less than MAX")
    if not 0.0 <= args.invalid_rate <= 1.0:
        parser.error("invalid-rate must be between 0.0 and 1.0")

    num_range = (args.num_range[0], args.num_range[1])

    # Generate to temp files to avoid persisting intermediate corpora
    with tempfile.TemporaryDirectory() as tmpdir:
        error_path = os.path.join(tmpdir, "instruction_corpus_error.txt")
        correct_path = os.path.join(tmpdir, "instruction_corpus_correct.txt")

        _generate_instruction_corpus(
            num_samples=args.num_samples,
            max_depth=args.max_depth,
            num_range=num_range,
            invalid_rate=args.invalid_rate,
            output_path=error_path,
        )

        _generate_instruction_corpus(
            num_samples=args.num_samples,
            max_depth=args.max_depth,
            num_range=num_range,
            invalid_rate=0.0,
            output_path=correct_path,
        )

        lines = _read_lines(error_path) + _read_lines(correct_path)

    # Shuffle and write mixed
    if args.seed is not None:
        random.seed(args.seed)
    random.shuffle(lines)
    _write_lines(args.output_mixed, lines)


if __name__ == "__main__":
    main()


