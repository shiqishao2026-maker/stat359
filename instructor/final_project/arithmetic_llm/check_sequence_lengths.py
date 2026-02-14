#!/usr/bin/env python3
"""Check actual sequence lengths in the corpus to determine optimal max_seq_length."""

import argparse
import json
import numpy as np
from .arithmetic_tokenizer import ArithmeticBPETokenizer


def analyze_corpus_lengths(corpus_path, tokenizer_path, max_samples=None, corpus_type='foundational'):
    """Analyze sequence lengths in a corpus.
    
    Args:
        corpus_path: Path to corpus file
        tokenizer_path: Path to tokenizer directory
        max_samples: Maximum number of samples to analyze (None = all)
        corpus_type: Type of corpus ('foundational' or 'instruction')
    """
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = ArithmeticBPETokenizer()
    tokenizer.load(tokenizer_path)
    
    print(f"\nAnalyzing corpus: {corpus_path}")
    print(f"Corpus type: {corpus_type}")
    if max_samples:
        print(f"Sampling: first {max_samples} lines")
    else:
        print("Analyzing: all lines")
    
    lengths = []
    skipped = 0
    
    with open(corpus_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                
                # Handle different corpus formats
                if corpus_type == 'foundational':
                    text = entry['problem'] + ' ' + entry['solution']
                elif corpus_type == 'instruction':
                    # Instruction format: problem + "<think>" prompt + solution
                    if 'problem' in entry and 'solution' in entry:
                        prompt = entry['problem'] + ' <think>'
                        text = prompt + ' ' + entry['solution']
                    else:
                        # Fallback for alternative schemas
                        text = entry.get('prompt', '') + ' ' + entry.get('response', '')
                else:
                    # Generic: concatenate all text fields
                    text = ' '.join(str(v) for v in entry.values() if isinstance(v, str))
                
                tokens = tokenizer.encode(text, add_special_tokens=True)
                lengths.append(len(tokens))
                
            except Exception:
                skipped += 1
                continue
    
    if not lengths:
        print("\nERROR: No valid sequences found!")
        return
    
    # Calculate statistics
    lengths_array = np.array(lengths)
    percentiles = [50, 75, 90, 95, 99, 99.5, 100]
    
    print("\n" + "=" * 60)
    print("SEQUENCE LENGTH ANALYSIS")
    print("=" * 60)
    print(f"\nTotal sequences analyzed: {len(lengths)}")
    if skipped > 0:
        print(f"Skipped (parse errors): {skipped}")
    
    print("\nBasic Statistics:")
    print(f"  Min length:     {np.min(lengths_array):>6} tokens")
    print(f"  Max length:     {np.max(lengths_array):>6} tokens")
    print(f"  Mean length:    {np.mean(lengths_array):>6.1f} tokens")
    print(f"  Median length:  {np.median(lengths_array):>6.1f} tokens")
    print(f"  Std deviation:  {np.std(lengths_array):>6.1f} tokens")
    
    print("\nPercentiles:")
    for p in percentiles:
        val = np.percentile(lengths_array, p)
        print(f"  {p:>5.1f}th percentile: {val:>6.0f} tokens")
    
    print("\nCoverage by max_seq_length:")
    thresholds = [64, 128, 192, 256, 384, 512, 768, 1024]
    for threshold in thresholds:
        count = np.sum(lengths_array <= threshold)
        pct = count / len(lengths_array) * 100
        truncated = len(lengths_array) - count
        print(f"  max_seq_length={threshold:>4}: {count:>6}/{len(lengths_array)} ({pct:>5.1f}%) | {truncated:>5} truncated")
    
    print("\nRecommendations:")
    # Find threshold that covers 95% and 99%
    p95 = np.percentile(lengths_array, 95)
    p99 = np.percentile(lengths_array, 99)
    
    print(f"  • For 95% coverage: max_seq_length >= {int(np.ceil(p95))}")
    print(f"  • For 99% coverage: max_seq_length >= {int(np.ceil(p99))}")
    print(f"  • For 100% coverage: max_seq_length >= {int(np.max(lengths_array))}")
    
    # Suggest practical values
    practical_95 = min([t for t in thresholds if t >= p95], default=int(np.ceil(p95)))
    practical_99 = min([t for t in thresholds if t >= p99], default=int(np.ceil(p99)))
    
    print("\n  Practical suggestions:")
    print(f"  • Balanced (95% coverage): --max-seq-length {practical_95}")
    print(f"  • Conservative (99% coverage): --max-seq-length {practical_99}")
    
    if np.max(lengths_array) > 1024:
        print("\n  ⚠️  Warning: Some sequences exceed 1024 tokens!")
        print("     Consider filtering or splitting long sequences.")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze corpus sequence lengths to determine optimal max_seq_length"
    )
    
    parser.add_argument(
        "--corpus-path",
        type=str,
        required=True,
        help="Path to corpus file to analyze"
    )
    
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path to trained tokenizer directory"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to analyze (default: all)"
    )
    
    parser.add_argument(
        "--corpus-type",
        type=str,
        choices=['foundational', 'instruction', 'auto'],
        default='auto',
        help="Type of corpus format (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect corpus type if needed
    corpus_type = args.corpus_type
    if corpus_type == 'auto':
        if 'instruction' in args.corpus_path.lower():
            corpus_type = 'instruction'
        else:
            corpus_type = 'foundational'
        print(f"Auto-detected corpus type: {corpus_type}")
    
    analyze_corpus_lengths(
        corpus_path=args.corpus_path,
        tokenizer_path=args.tokenizer_path,
        max_samples=args.max_samples,
        corpus_type=corpus_type
    )


if __name__ == "__main__":
    main()


