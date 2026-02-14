"""Simple script to print the token table in a clean format."""

import os
import sys
from .arithmetic_tokenizer import ArithmeticBPETokenizer


def print_token_table(tokenizer, format='table', max_tokens=None):
    """Print token table in specified format.
    
    Args:
        tokenizer: Tokenizer instance
        format: 'table', 'csv', or 'json'
        max_tokens: Maximum number of tokens to show (None for all)
    """
    sorted_tokens = sorted(tokenizer.token2id.items(), key=lambda x: x[1])
    
    if max_tokens:
        sorted_tokens = sorted_tokens[:max_tokens]
    
    if format == 'table':
        print(f"{'ID':<6} | {'Token':<30} | {'Type':<15}")
        print("-" * 60)
        
        for token, token_id in sorted_tokens:
            # Determine type
            if token in tokenizer.special_tokens:
                token_type = "Special"
            elif hasattr(tokenizer, 'atomic_symbols') and token in tokenizer.atomic_symbols:
                token_type = "Atomic Symbol"
            elif token.replace('</w>', '').isdigit():
                token_type = "Digit"
            elif '</w>' in token:
                token_type = "Word"
            elif len(token) == 1:
                token_type = "Char"
            else:
                token_type = "Subword"
            
            # Escape for display
            display_token = repr(token)[1:-1]  # Remove outer quotes
            print(f"{token_id:<6} | {display_token:<30} | {token_type:<15}")
    
    elif format == 'csv':
        print("ID,Token,Type")
        for token, token_id in sorted_tokens:
            # Determine type
            if token in tokenizer.special_tokens:
                token_type = "Special"
            elif hasattr(tokenizer, 'atomic_symbols') and token in tokenizer.atomic_symbols:
                token_type = "Atomic Symbol"
            elif token.replace('</w>', '').isdigit():
                token_type = "Digit"
            elif '</w>' in token:
                token_type = "Word"
            elif len(token) == 1:
                token_type = "Char"
            else:
                token_type = "Subword"
            
            # Escape for CSV
            token_escaped = token.replace('"', '""')
            print(f'{token_id},"{token_escaped}",{token_type}')
    
    elif format == 'json':
        import json
        tokens_list = []
        for token, token_id in sorted_tokens:
            # Determine type
            if token in tokenizer.special_tokens:
                token_type = "Special"
            elif hasattr(tokenizer, 'atomic_symbols') and token in tokenizer.atomic_symbols:
                token_type = "Atomic Symbol"
            elif token.replace('</w>', '').isdigit():
                token_type = "Digit"
            elif '</w>' in token:
                token_type = "Word"
            elif len(token) == 1:
                token_type = "Char"
            else:
                token_type = "Subword"
            
            tokens_list.append({
                'id': token_id,
                'token': token,
                'type': token_type
            })
        
        print(json.dumps(tokens_list, indent=2, ensure_ascii=False))


def main():
    """Main function."""
    # Parse arguments
    format = 'table'
    max_tokens = None
    tokenizer_path = "data/tokenizer/tokenizer.pkl"

    # Parse command-line arguments
    arg_idx = 1
    if len(sys.argv) > arg_idx:
        if sys.argv[arg_idx] in ['table', 'csv', 'json']:
            format = sys.argv[arg_idx]
            arg_idx += 1
        elif sys.argv[arg_idx].isdigit():
            max_tokens = int(sys.argv[arg_idx])
            arg_idx += 1

    if len(sys.argv) > arg_idx:
        if sys.argv[arg_idx].isdigit():
            max_tokens = int(sys.argv[arg_idx])
            arg_idx += 1
        else:
            tokenizer_path = sys.argv[arg_idx]
            arg_idx += 1

    if len(sys.argv) > arg_idx:
        # If both max_tokens and tokenizer_path are provided
        if sys.argv[arg_idx].isdigit():
            max_tokens = int(sys.argv[arg_idx])
        else:
            tokenizer_path = sys.argv[arg_idx]

    # Load tokenizer
    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at {tokenizer_path}", file=sys.stderr)
        print("Please run: python homeowrk/arithmetic_llm/train_tokenizer.py", file=sys.stderr)
        sys.exit(1)

    tokenizer = ArithmeticBPETokenizer(vocab_size=1000)
    tokenizer_dir = os.path.dirname(tokenizer_path)
    tokenizer.load(tokenizer_dir)

    # Print header (only for table format)
    if format == 'table':
        print(f"Token Table (Vocabulary Size: {len(tokenizer.token2id)})")
        if max_tokens:
            print(f"Showing first {max_tokens} tokens")
        print()

    # Print token table
    print_token_table(tokenizer, format=format, max_tokens=max_tokens)

    # Print footer (only for table format)
    if format == 'table':
        print()
        if max_tokens and max_tokens < len(tokenizer.token2id):
            print(f"... and {len(tokenizer.token2id) - max_tokens} more tokens")
        print(f"\nTotal: {len(tokenizer.token2id)} tokens")


if __name__ == "__main__":
        if '--help' in sys.argv or '-h' in sys.argv:
                print("""
Usage: python print_token_table.py [format] [max_tokens] [tokenizer_path]

Arguments:
    format         Output format: 'table' (default), 'csv', or 'json'
    max_tokens     Maximum number of tokens to show (default: all)
    tokenizer_path Path to tokenizer.pkl (default: data/tokenizer/tokenizer.pkl)

Examples:
    python print_token_table.py                        # Show all tokens as table
    python print_token_table.py 50                     # Show first 50 tokens as table
    python print_token_table.py csv                    # Show all tokens as CSV
    python print_token_table.py json 20                # Show first 20 tokens as JSON
    python print_token_table.py table 100              # Show first 100 tokens as table
    python print_token_table.py table 100 mytokenizer/tokenizer.pkl  # Custom tokenizer path

Output can be redirected to a file:
    python print_token_table.py csv > tokens.csv
    python print_token_table.py json > tokens.json
""")
                sys.exit(0)

        main()


