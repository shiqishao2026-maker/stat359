#!/usr/bin/env python3
"""Test script to verify EOS token truncation behavior."""

from .arithmetic_tokenizer import ArithmeticBPETokenizer


def test_eos_truncation():
    """Test that decode properly truncates at first EOS token."""
    
    # Create a mock tokenizer with minimal setup
    tokenizer = ArithmeticBPETokenizer()
    
    # Manually set up token mappings for testing
    tokenizer.token2id = {
        '<pad>': 0,
        '<unk>': 1,
        '<bos>': 2,
        '<eos>': 3,
        '<think>': 4,
        '</think>': 5,
        'Evaluate:': 6,
        '5': 7,
        '+': 8,
        '3': 9,
        '=': 10,
        '8': 11,
        'Step': 12,
        '1:': 13,
        'Final': 14,
        'Result:': 15,
        'garbage': 16,
        'after': 17,
        'eos': 18,
    }
    
    tokenizer.id2token = {v: k for k, v in tokenizer.token2id.items()}
    tokenizer.special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>', '<think>', '</think>']
    
    print("Testing EOS truncation behavior\n")
    print("=" * 60)
    
    # Test 1: Normal sequence with EOS at end
    print("\nTest 1: Normal sequence with EOS at end")
    token_ids = [2, 6, 7, 8, 9, 3]  # <bos> Evaluate: 5 + 3 <eos>
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    print(f"  Input IDs: {token_ids}")
    print(f"  Decoded: '{decoded}'")
    print("  ✓ Should not contain garbage after EOS")
    
    # Test 2: Sequence with garbage after EOS (should be truncated)
    print("\nTest 2: Sequence with garbage after EOS (should be truncated)")
    token_ids = [2, 6, 7, 8, 9, 3, 16, 17, 18]  # <bos> Evaluate: 5 + 3 <eos> garbage after eos
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    print(f"  Input IDs: {token_ids}")
    print(f"  Decoded: '{decoded}'")
    assert 'garbage' not in decoded.lower(), "ERROR: Garbage after EOS was not truncated!"
    print("  ✓ Correctly truncated at first EOS")
    
    # Test 3: Sequence without EOS
    print("\nTest 3: Sequence without EOS")
    token_ids = [2, 6, 7, 8, 9]  # <bos> Evaluate: 5 + 3
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    print(f"  Input IDs: {token_ids}")
    print(f"  Decoded: '{decoded}'")
    print("  ✓ Decoded entire sequence")
    
    # Test 4: Multiple EOS tokens (should stop at first)
    print("\nTest 4: Multiple EOS tokens (should stop at first)")
    token_ids = [2, 6, 7, 3, 8, 9, 3]  # <bos> Evaluate: 5 <eos> + 3 <eos>
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    print(f"  Input IDs: {token_ids}")
    print(f"  Decoded: '{decoded}'")
    # Should only have "Evaluate: 5", not "+ 3"
    assert '+' not in decoded, "ERROR: Did not stop at first EOS!"
    print("  ✓ Correctly stopped at first EOS")
    
    # Test 5: With skip_special_tokens=False
    print("\nTest 5: With skip_special_tokens=False (EOS visible)")
    token_ids = [2, 6, 7, 8, 9, 3, 16]  # <bos> Evaluate: 5 + 3 <eos> garbage
    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    print(f"  Input IDs: {token_ids}")
    print(f"  Decoded: '{decoded}'")
    assert '<eos>' in decoded, "ERROR: EOS token not visible!"
    assert 'garbage' not in decoded.lower(), "ERROR: Garbage after EOS was not truncated!"
    print("  ✓ EOS visible but still truncated after it")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("\nSummary:")
    print("  - decode() truncates at first EOS token")
    print("  - Anything after EOS is discarded")
    print("  - Works with skip_special_tokens=True and False")
    print("  - Handles sequences without EOS gracefully")


if __name__ == "__main__":
    test_eos_truncation()


