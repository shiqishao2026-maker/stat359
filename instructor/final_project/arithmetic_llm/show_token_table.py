"""Display the tokenizer vocabulary (token table) in various formats."""

import os
from .arithmetic_tokenizer import ArithmeticBPETokenizer


def print_separator(title=""):
    """Print a visual separator."""
    print("\n" + "=" * 80)
    if title:
        print(f" {title}")
        print("=" * 80)


def show_full_token_table(tokenizer, max_tokens=None):
    """Show the complete token table."""
    print_separator("COMPLETE TOKEN TABLE")
    
    print(f"\nVocabulary size: {len(tokenizer.token2id)}")
    print(f"\nShowing {'all' if max_tokens is None else f'first {max_tokens}'} tokens:\n")
    
    # Sort by token ID
    sorted_tokens = sorted(tokenizer.token2id.items(), key=lambda x: x[1])
    
    if max_tokens:
        sorted_tokens = sorted_tokens[:max_tokens]
    
    print(f"{'Token ID':<10} | {'Token':<30} | {'Type'}")
    print("-" * 80)
    
    for token, token_id in sorted_tokens:
        # Determine token type
        if token in tokenizer.special_tokens:
            token_type = "Special"
        elif hasattr(tokenizer, 'atomic_symbols') and token in tokenizer.atomic_symbols:
            token_type = "Atomic Symbol"
        elif token.replace('</w>', '').isdigit():
            token_type = "Digit"
        elif '</w>' in token:
            token_type = "Word"
        elif len(token) == 1:
            token_type = "Character"
        else:
            token_type = "Subword"
        
        # Escape special characters for display
        display_token = token.replace('\n', '\\n').replace('\t', '\\t')
        
        print(f"{token_id:<10} | {display_token:<30} | {token_type}")


def show_token_table_by_category(tokenizer):
    """Show token table organized by category."""
    print_separator("TOKEN TABLE BY CATEGORY")
    
    # Categorize tokens
    special_tokens = []
    atomic_symbols = []
    digits = []
    operators = []
    words = []
    subwords = []
    characters = []
    
    for token, token_id in tokenizer.token2id.items():
        if token in tokenizer.special_tokens:
            special_tokens.append((token, token_id))
        elif hasattr(tokenizer, 'atomic_symbols') and token in tokenizer.atomic_symbols:
            atomic_symbols.append((token, token_id))
        elif token.replace('</w>', '').isdigit():
            digits.append((token, token_id))
        elif token in '+-()':
            operators.append((token, token_id))
        elif '</w>' in token and len(token.replace('</w>', '')) > 1:
            words.append((token, token_id))
        elif len(token) == 1:
            characters.append((token, token_id))
        else:
            subwords.append((token, token_id))
    
    # Display each category
    categories = [
        ("Special Tokens", special_tokens),
        ("Atomic Symbols", atomic_symbols),
        ("Digits", digits),
        ("Operators", operators),
        ("Complete Words", words),
        ("Subwords", subwords),
        ("Single Characters", characters),
    ]
    
    for category_name, tokens in categories:
        if tokens:
            print(f"\n{category_name} ({len(tokens)} tokens):")
            print(f"{'Token ID':<10} | {'Token':<30}")
            print("-" * 45)
            
            # Sort by token ID
            tokens.sort(key=lambda x: x[1])
            
            # Show first 20 of each category
            for token, token_id in tokens[:20]:
                display_token = token.replace('\n', '\\n').replace('\t', '\\t')
                print(f"{token_id:<10} | {display_token:<30}")
            
            if len(tokens) > 20:
                print(f"... and {len(tokens) - 20} more")


def show_token_statistics(tokenizer):
    """Show statistics about the token table."""
    print_separator("TOKEN TABLE STATISTICS")
    
    # Count different types
    special_count = sum(1 for t in tokenizer.token2id if t in tokenizer.special_tokens)
    atomic_count = sum(1 for t in tokenizer.token2id if hasattr(tokenizer, 'atomic_symbols') and t in tokenizer.atomic_symbols)
    digit_count = sum(1 for t in tokenizer.token2id if t.replace('</w>', '').isdigit())
    word_count = sum(1 for t in tokenizer.token2id if '</w>' in t and len(t.replace('</w>', '')) > 1)
    single_char_count = sum(1 for t in tokenizer.token2id if len(t) == 1)
    
    print(f"\nVocabulary size: {len(tokenizer.token2id)}")
    print("\nToken type breakdown:")
    print(f"  Special tokens:      {special_count:4d}")
    print(f"  Atomic symbols:      {atomic_count:4d}")
    print(f"  Digit tokens:        {digit_count:4d}")
    print(f"  Complete words:      {word_count:4d}")
    print(f"  Single characters:   {single_char_count:4d}")
    print(f"  Other (subwords):    {len(tokenizer.token2id) - special_count - atomic_count - digit_count - word_count - single_char_count:4d}")


def show_bpe_merges(tokenizer, max_merges=20):
    """Show BPE merge operations."""
    print_separator("BPE MERGE OPERATIONS")
    
    print(f"\nTotal BPE merges: {len(tokenizer.bpe_codes)}")
    print(f"\nShowing first {max_merges} merges (by rank):\n")
    
    print(f"{'Rank':<6} | {'Left':<15} | {'Right':<15} | {'Merged':<20}")
    print("-" * 70)
    
    # Sort by rank (merge order)
    sorted_merges = sorted(tokenizer.bpe_codes.items(), key=lambda x: x[1])
    
    for (left, right), rank in sorted_merges[:max_merges]:
        merged = left + right
        print(f"{rank:<6} | {left:<15} | {right:<15} | {merged:<20}")
    
    if len(sorted_merges) > max_merges:
        print(f"\n... and {len(sorted_merges) - max_merges} more merges")


def search_tokens(tokenizer, query):
    """Search for tokens containing a query string."""
    print_separator(f"SEARCH RESULTS FOR: '{query}'")
    
    matches = []
    for token, token_id in tokenizer.token2id.items():
        if query.lower() in token.lower():
            matches.append((token, token_id))
    
    if matches:
        print(f"\nFound {len(matches)} matching tokens:\n")
        print(f"{'Token ID':<10} | {'Token':<30}")
        print("-" * 45)
        
        matches.sort(key=lambda x: x[1])
        for token, token_id in matches:
            display_token = token.replace('\n', '\\n').replace('\t', '\\t')
            print(f"{token_id:<10} | {display_token:<30}")
    else:
        print(f"\nNo tokens found containing '{query}'")


def show_token_examples(tokenizer):
    """Show examples of how text is tokenized."""
    print_separator("TOKENIZATION EXAMPLES")
    
    examples = [
        "5 + 3",
        "((7 + 4))",
        "Evaluate: 10 - 5",
        "Step 1: 5 + 3 = 8",
        "Result: 42",
        "((((7 + 4) + (4 + 5)) - 12))",
    ]
    
    print("\nHow different texts are tokenized:\n")
    
    for text in examples:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        tokens = [tokenizer.id2token.get(tid) for tid in token_ids]
        
        print(f"Text:      {text}")
        print(f"Tokens:    {tokens}")
        print(f"Token IDs: {token_ids}")
        print(f"Count:     {len(tokens)} tokens")
        print()


def export_token_table_csv(tokenizer, output_path="token_table.csv"):
    """Export token table to CSV file."""
    print_separator("EXPORT TO CSV")
    
    import csv
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Token ID', 'Token', 'Type'])
        
        sorted_tokens = sorted(tokenizer.token2id.items(), key=lambda x: x[1])
        
        for token, token_id in sorted_tokens:
            # Determine token type
            if token in tokenizer.special_tokens:
                token_type = "Special"
            elif hasattr(tokenizer, 'atomic_symbols') and token in tokenizer.atomic_symbols:
                token_type = "Atomic Symbol"
            elif token.replace('</w>', '').isdigit():
                token_type = "Digit"
            elif '</w>' in token:
                token_type = "Word"
            elif len(token) == 1:
                token_type = "Character"
            else:
                token_type = "Subword"
            
            writer.writerow([token_id, token, token_type])
    
    print(f"\n✓ Token table exported to: {output_path}")
    print(f"  Total tokens: {len(tokenizer.token2id)}")


def main():
    """Main function."""
    print("=" * 80)
    print(" TOKEN TABLE VIEWER")
    print("=" * 80)
    
    # Check if tokenizer exists
    tokenizer_path = "data/tokenizer/tokenizer.pkl"
    
    if not os.path.exists(tokenizer_path):
        print(f"\nError: Tokenizer not found at {tokenizer_path}")
        print("Please run: python homeowrk/arithmetic_llm/train_tokenizer.py")
        return
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = ArithmeticBPETokenizer(vocab_size=1000)
    tokenizer_dir = os.path.dirname(tokenizer_path)
    tokenizer.load(tokenizer_dir)
    print(f"✓ Loaded tokenizer with {len(tokenizer.token2id)} tokens")
    
    # Show different views
    show_token_statistics(tokenizer)
    show_token_table_by_category(tokenizer)
    show_bpe_merges(tokenizer, max_merges=20)
    show_token_examples(tokenizer)
    
    # Show full table (limited)
    print_separator("FULL TOKEN TABLE (First 50)")
    show_full_token_table(tokenizer, max_tokens=50)
    
    # Search examples
    search_tokens(tokenizer, "Eval")
    search_tokens(tokenizer, "Step")
    
    # Export option
    export_token_table_csv(tokenizer, "token_table.csv")
    
    print("\n" + "=" * 80)
    print(" USAGE")
    print("=" * 80)
    print("""
To view the token table:
  python homeowrk/arithmetic_llm/show_token_table.py

To search for specific tokens:
  python -c "
  from .arithmetic_tokenizer import ArithmeticBPETokenizer
  tokenizer = ArithmeticBPETokenizer(vocab_size=1000)
  tokenizer.load('data/tokenizer')
  
  # Search for tokens
  query = 'Step'
  matches = [(t, tid) for t, tid in tokenizer.token2id.items() if query in t]
  for token, token_id in sorted(matches, key=lambda x: x[1]):
      print(f'{token_id}: {token}')
  "

To get a specific token:
  python -c "
  from .arithmetic_tokenizer import ArithmeticBPETokenizer
  tokenizer = ArithmeticBPETokenizer(vocab_size=1000)
  tokenizer.load('data/tokenizer')
  
  # Get token by ID
  print(tokenizer.id2token[42])
  
  # Get ID by token
  print(tokenizer.token2id['Step</w>'])
  "
""")
    print("=" * 80)


if __name__ == "__main__":
    main()


