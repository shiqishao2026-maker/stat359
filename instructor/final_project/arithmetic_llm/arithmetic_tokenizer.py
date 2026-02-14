"""BPE tokenizer module for arithmetic expressions."""

import re
import pickle
import json
from collections import Counter
from typing import List
from tqdm import tqdm


class ArithmeticBPETokenizer:
    """BPE tokenizer specialized for arithmetic expressions."""
    
    def __init__(self, vocab_size: int = 1000):
        """
        Initialize BPE tokenizer for arithmetic data.
        
        Args:
            vocab_size: Target vocabulary size
        """
        self.vocab_size = vocab_size
        self.bpe_codes = {}
        self.vocab = {}
        self.token2id = {}
        self.id2token = {}
        
        # Define special tokens for arithmetic reasoning
        self.special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>', '<think>', '</think>']
        
        # Define atomic symbols that should never be merged
        self.atomic_symbols = ['+', '-', '(', ')', ':']
    
    def _pre_tokenize(self, text: str) -> list:
        """Pre-tokenize text by splitting atomic symbols.
        
        This ensures operators and special symbols are always separate tokens
        before BPE processing.
        
        Args:
            text: Input text
            
        Returns:
            List of pre-tokenized words
        """
        # Add spaces around atomic symbols
        for symbol in self.atomic_symbols:
            text = text.replace(symbol, f' {symbol} ')
        
        # Split on whitespace and filter empty strings
        return [word for word in text.split() if word]
    
    def _get_stats(self, corpus: dict) -> Counter:
        """Get frequency statistics of adjacent token pairs.
        
        Excludes pairs involving atomic symbols (+, -, (, ), :) to prevent merging.
        """
        # Symbols that should never be merged
        atomic_symbols = {'+', '-', '(', ')', ':'}
        
        pairs = Counter()
        for word, freq in corpus.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                left, right = symbols[i], symbols[i + 1]
                
                # Skip pairs where either token is an atomic symbol
                # (or contains only atomic symbols)
                left_clean = left.replace('</w>', '')
                right_clean = right.replace('</w>', '')
                
                if left_clean in atomic_symbols or right_clean in atomic_symbols:
                    continue
                
                pairs[(left, right)] += freq
        return pairs
    
    def _merge_vocab(self, pair: tuple, corpus: dict) -> dict:
        """Merge the most frequent pair in the vocabulary."""
        pattern = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + pattern + r'(?!\S)')
        new_corpus = {}
        for word in corpus:
            new_word = pattern.sub(''.join(pair), word)
            new_corpus[new_word] = corpus[word]
        return new_corpus
    
    def train(self, corpus_path: str) -> None:
        """
        Train tokenizer on corpus.
        
        Args:
            corpus_path: Path to training corpus file (JSONL format)
        """
        # Build initial corpus with character-level tokens
        corpus = Counter()
        
        with open(corpus_path, 'r') as f:
            for line in tqdm(f, desc="Building corpus"):
                line = line.strip()
                if not line:
                    continue
                
                # Parse JSONL format
                try:
                    data = json.loads(line)
                    # Extract problem and solution fields
                    problem = data.get('problem', '')
                    solution = data.get('solution', '')
                    # Join with newline
                    text = problem + '\n' + solution
                except json.JSONDecodeError:
                    # If not valid JSON, treat as plain text (backward compatibility)
                    text = line
                
                # Remove special tokens from the text entirely before processing
                # This prevents BPE from learning any subword combinations of them
                for special_token in self.special_tokens:
                    text = text.replace(special_token, ' ')
                
                # Pre-tokenize to split atomic symbols
                words = self._pre_tokenize(text)
                
                for word in words:
                    # Skip empty words (from removed special tokens)
                    if not word:
                        continue
                    # Keep atomic symbols as single tokens
                    if word in self.atomic_symbols:
                        corpus[word] += 1
                    else:
                        # Add character-level representation with end-of-word marker
                        corpus[' '.join(list(word)) + ' </w>'] += 1
        
        self.vocab = dict(corpus)
        
        # Perform BPE merges
        for i in tqdm(range(self.vocab_size), desc="BPE merges"):
            pairs = self._get_stats(self.vocab)
            if not pairs:
                break
            best = pairs.most_common(1)[0][0]
            self.vocab = self._merge_vocab(best, self.vocab)
            self.bpe_codes[best] = i
        
        # Build token vocabulary from final vocab state
        tokens = set()
        for word in self.vocab:
            tokens.update(word.split())
        
        # IMPORTANT: Also add all intermediate merge results to vocabulary
        # This ensures that partial BPE merges during encoding have valid tokens
        for (left, right), _ in self.bpe_codes.items():
            merged = left + right
            tokens.add(merged)
            # Also add the individual parts
            tokens.add(left)
            tokens.add(right)
        
        # Ensure special tokens are always in the vocabulary
        tokens.update(self.special_tokens)
        
        # Ensure atomic symbols are always in the vocabulary
        tokens.update(self.atomic_symbols)
        
        # Ensure all individual characters that appear in arithmetic are in vocabulary
        # This includes digits and the end-of-word marker
        arithmetic_chars = set('0123456789 ')
        tokens.update(arithmetic_chars)
        tokens.add('</w>')  # Add end-of-word marker
        
        # Add all ASCII letters (a-z, A-Z) to handle any text
        # This ensures unknown words can still be encoded character-by-character
        for i in range(ord('a'), ord('z') + 1):
            tokens.add(chr(i))
        for i in range(ord('A'), ord('Z') + 1):
            tokens.add(chr(i))
        
        # Add common punctuation that might appear
        common_punctuation = '.,!?;:\'"'
        tokens.update(common_punctuation)
        
        # Ensure single digits with </w> marker are in vocabulary
        for digit in '0123456789':
            tokens.add(digit + '</w>')
        
        # Ensure atomic symbols with </w> marker are in vocabulary
        for symbol in self.atomic_symbols:
            tokens.add(symbol + '</w>')
        
        # Create token-to-ID mappings
        self.token2id = {token: idx for idx, token in enumerate(sorted(tokens))}
        self.id2token = {idx: token for token, idx in self.token2id.items()}
    
    def save(self, save_dir: str) -> None:
        """
        Save tokenizer vocabulary and merges.
        
        Args:
            save_dir: Directory path to save tokenizer files
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        filepath = os.path.join(save_dir, 'tokenizer.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vocab_size': self.vocab_size,
                'bpe_codes': self.bpe_codes,
                'vocab': self.vocab,
                'token2id': self.token2id,
                'id2token': self.id2token,
                'special_tokens': self.special_tokens,
                'atomic_symbols': self.atomic_symbols,
            }, f)
    
    def load(self, save_dir: str) -> None:
        """
        Load saved tokenizer.
        
        Args:
            save_dir: Directory path containing tokenizer files
        """
        import os
        filepath = os.path.join(save_dir, 'tokenizer.pkl')
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.vocab_size = state['vocab_size']
        self.bpe_codes = state['bpe_codes']
        self.vocab = state['vocab']
        self.token2id = state['token2id']
        self.id2token = state['id2token']
        self.special_tokens = state.get('special_tokens', self.special_tokens)
        self.atomic_symbols = state.get('atomic_symbols', ['+', '-', '(', ')', ':'])
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add <bos> and <eos> tokens (default: True)
            
        Returns:
            List of token IDs
        """
        if not text:
            if add_special_tokens:
                return [self.token2id.get('<bos>', 0), self.token2id.get('<eos>', 0)]
            return []
        
        tokens = []
        
        # Pre-tokenize to split atomic symbols
        words = self._pre_tokenize(text)
        
        for word in words:
            # Check if word is a special token
            if word in self.special_tokens:
                tokens.append(word)
                continue
            
            # Check if word is an atomic symbol
            if word in self.atomic_symbols:
                tokens.append(word)
                continue
            
            # Check if word is a single digit (treat as complete word with </w>)
            if len(word) == 1 and word.isdigit():
                tokens.append(word + '</w>')
                continue
            
            # Apply BPE encoding for multi-character words
            word_chars = list(word) + ['</w>']
            
            while True:
                pairs = [(word_chars[i], word_chars[i+1]) for i in range(len(word_chars)-1)]
                # Only consider pairs that are in BPE codes
                pair_ranks = {}
                for pair in pairs:
                    if pair in self.bpe_codes:
                        pair_ranks[pair] = self.bpe_codes[pair]
                
                if not pair_ranks:
                    break
                
                best_pair = min(pair_ranks, key=pair_ranks.get)
                
                # Merge the best pair
                i = 0
                new_word = []
                while i < len(word_chars):
                    if i < len(word_chars) - 1 and (word_chars[i], word_chars[i+1]) == best_pair:
                        new_word.append(word_chars[i] + word_chars[i+1])
                        i += 2
                    else:
                        new_word.append(word_chars[i])
                        i += 1
                word_chars = new_word
            
            tokens.extend(word_chars)
        
        # Convert tokens to IDs
        token_ids = [self.token2id.get(token, self.token2id.get('<unk>', 0)) for token in tokens]
        
        # Add BOS and EOS tokens if requested
        if add_special_tokens:
            bos_id = self.token2id.get('<bos>', 0)
            eos_id = self.token2id.get('<eos>', 0)
            token_ids = [bos_id] + token_ids + [eos_id]
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens like <bos>, <eos>, <pad>
            
        Returns:
            Decoded text string
        """
        # First, truncate at EOS token if present
        eos_id = self.token2id.get('<eos>', None)
        if eos_id is not None and eos_id in token_ids:
            # Find first occurrence of EOS and truncate
            eos_idx = token_ids.index(eos_id)
            token_ids = token_ids[:eos_idx + 1]  # Include EOS for now
        
        tokens = [self.id2token.get(i, '<unk>') for i in token_ids]
        
        # Filter out special tokens if requested
        if skip_special_tokens:
            special_to_skip = {'<bos>', '<eos>', '<pad>', '<unk>'}
            tokens = [t for t in tokens if t not in special_to_skip]
        
        text = ''
        word = ''
        
        for token in tokens:
            # Handle special tokens (that weren't filtered)
            if token in self.special_tokens:
                if word:
                    text += word + ' '
                    word = ''
                text += token + ' '
            # Handle atomic symbols
            elif token in self.atomic_symbols:
                if word:
                    text += word + ' '
                    word = ''
                text += token + ' '
            # Handle end-of-word marker
            elif token.endswith('</w>'):
                word += token[:-4]  # Remove '</w>'
                text += word + ' '
                word = ''
            # Handle single digit characters
            elif len(token) == 1 and token.isdigit():
                # Accumulate digits that are part of a multi-digit number
                word += token
            else:
                # Accumulate other characters
                word += token
        
        if word:
            text += word
        
        return text.strip()
