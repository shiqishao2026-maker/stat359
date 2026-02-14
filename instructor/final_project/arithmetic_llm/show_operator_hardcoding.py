"""Script to highlight the exact lines where operators are hardcoded."""

import os


def print_code_section(title, lines_with_numbers):
    """Print a code section with line numbers."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)
    print()
    for line_num, line in lines_with_numbers:
        print(f"{line_num:4d} | {line}")
    print()


def main():
    print("=" * 80)
    print(" OPERATOR HARDCODING IN arithmetic_tokenizer.py")
    print("=" * 80)
    
    # Read the tokenizer file
    tokenizer_file = "arithmetic_tokenizer.py"
    
    if not os.path.exists(tokenizer_file):
        print(f"Error: {tokenizer_file} not found")
        return
    
    with open(tokenizer_file, 'r') as f:
        lines = f.readlines()
    
    # Section 1: Training - Adding operators to vocabulary
    print_code_section(
        "LOCATION 1: train() method - Lines 103-113",
        [
            (103, "        # Ensure all individual characters that appear in arithmetic are in vocabulary"),
            (104, "        # This includes digits, operators, parentheses, and the end-of-word marker"),
            (105, "        arithmetic_chars = set('0123456789+-() ')  # ← HARDCODED HERE"),
            (106, "        tokens.update(arithmetic_chars)"),
            (107, "        tokens.add('</w>')  # Add end-of-word marker"),
            (108, "        "),
            (109, "        # Ensure single digits with </w> marker are in vocabulary"),
            (110, "        for digit in '0123456789':"),
            (111, "            tokens.add(digit + '</w>')"),
            (112, "        "),
            (113, "        # Ensure operators and parentheses with </w> marker are in vocabulary"),
            (114, "        for char in '+-()':  # ← HARDCODED HERE"),
            (115, "            tokens.add(char + '</w>')"),
        ]
    )
    
    # Section 2: Encoding - Skipping BPE for operators
    print_code_section(
        "LOCATION 2: encode() method - Lines 189-192",
        [
            (189, "            # Check if word is a single operator or parenthesis (no </w> needed)"),
            (190, "            if len(word) == 1 and word in '+-()':  # ← HARDCODED HERE"),
            (191, "                tokens.append(word)"),
            (192, "                continue  # ← SKIPS BPE PROCESSING"),
        ]
    )
    
    print("=" * 80)
    print(" EXPLANATION")
    print("=" * 80)
    print("""
The operators +, -, (, ) are hardcoded in TWO places:

1. DURING TRAINING (lines 105 and 114):
   - Line 105: arithmetic_chars = set('0123456789+-() ')
   - Line 114: for char in '+-()':
   
   These lines FORCE the operators to be in the vocabulary, regardless of
   whether they appear in the training corpus.

2. DURING ENCODING (line 190):
   - Line 190: if len(word) == 1 and word in '+-()':
   - Line 192: continue
   
   This check ensures that when encoding text, operators are kept as single
   tokens and SKIP the BPE merging algorithm entirely.

RESULT:
- Operators are ALWAYS single tokens
- Operators are ALWAYS in the vocabulary
- Operators NEVER get merged with other characters
- Consistent tokenization across all inputs

The string '+-()' appears exactly 3 times in the code:
1. Line 105: Adding to vocabulary during training
2. Line 114: Adding with </w> marker during training
3. Line 190: Checking during encoding to skip BPE
""")
    print("=" * 80)
    
    # Show actual code from file
    print("\n" + "=" * 80)
    print(" ACTUAL CODE FROM FILE")
    print("=" * 80)
    
    print("\n--- Lines 103-115 (train method) ---")
    for i in range(102, 115):
        if i < len(lines):
            print(f"{i+1:4d} | {lines[i]}", end='')
    
    print("\n--- Lines 189-192 (encode method) ---")
    for i in range(188, 192):
        if i < len(lines):
            print(f"{i+1:4d} | {lines[i]}", end='')
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
