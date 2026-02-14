import re
from collections import Counter, defaultdict
import pickle
from tqdm import tqdm  # Added tqdm import

class BPETokenizer:
    def __init__(self, vocab_size=10000, special_tokens=None):
        self.vocab_size = vocab_size
        self.bpe_codes = {}
        self.vocab = {}
        self.token2id = {}
        self.id2token = {}
        # Define default special tokens for GPT/instruction tuning
        if special_tokens is None:
            self.special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>', '<user>', '<assistant>', '<system>']
        else:
            self.special_tokens = special_tokens

    def get_stats(self, corpus):
        pairs = Counter()
        for word, freq in corpus.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_vocab(self, pair, corpus):
        pattern = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + pattern + r'(?!\S)')
        new_corpus = {}
        for word in corpus:
            new_word = pattern.sub(''.join(pair), word)
            new_corpus[new_word] = corpus[word]
        return new_corpus

    def fit(self, texts):
        corpus = Counter()
        for line in tqdm(texts, desc="Building corpus"):  # tqdm for progress
            for word in line.strip().split():
                corpus[' '.join(list(word)) + ' </w>'] += 1
        self.vocab = dict(corpus)
        for i in tqdm(range(self.vocab_size), desc="BPE merges"):  # tqdm for progress
            pairs = self.get_stats(self.vocab)
            if not pairs:
                break
            best = pairs.most_common(1)[0][0]
            self.vocab = self.merge_vocab(best, self.vocab)
            self.bpe_codes[best] = i
        tokens = set()
        for word in self.vocab:
            tokens.update(word.split())
        # Ensure special tokens are always in the vocabulary
        tokens.update(self.special_tokens)
        self.token2id = {token: idx for idx, token in enumerate(sorted(tokens))}
        self.id2token = {idx: token for token, idx in self.token2id.items()}

    def encode(self, text, add_special_tokens=False):
        tokens = []
        if add_special_tokens:
            tokens.append('<bos>')
        for word in text.strip().split():
            # Check if word is a special token
            if word in self.special_tokens:
                tokens.append(word)
                continue
            word = list(word) + ['</w>']
            while True:
                pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
                pair_ranks = {pair: self.bpe_codes.get(pair, float('inf')) for pair in pairs}
                if not pair_ranks:
                    break
                best_pair = min(pair_ranks, key=pair_ranks.get)
                if best_pair not in self.bpe_codes:
                    break
                i = 0
                new_word = []
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
                        new_word.append(word[i] + word[i+1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                word = new_word
            tokens.extend(word)
        if add_special_tokens:
            tokens.append('<eos>')
        return [self.token2id.get(token, self.token2id.get('<unk>', 0)) for token in tokens]

    def decode(self, token_ids, remove_special_tokens=True):
        tokens = [self.id2token.get(i, '<unk>') for i in token_ids]
        if remove_special_tokens:
            tokens = [t for t in tokens if t not in self.special_tokens or t == '</w>']
        text = ''
        word = ''
        for token in tokens:
            if token.endswith('</w>'):
                word += token[:-4]  # Remove '</w>'
                text += word + ' '
                word = ''
            else:
                word += token
        if word:
            text += word  # Add any remaining word
        return text.strip()

    def save(self, filepath):
        """Save the tokenizer state to a file using pickle."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vocab_size': self.vocab_size,
                'bpe_codes': self.bpe_codes,
                'vocab': self.vocab,
                'token2id': self.token2id,
                'id2token': self.id2token,
                'special_tokens': self.special_tokens,
            }, f)

    @classmethod
    def load(cls, filepath):
        """Load the tokenizer state from a file using pickle."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        tokenizer = cls(vocab_size=state['vocab_size'], special_tokens=state.get('special_tokens'))
        tokenizer.bpe_codes = state['bpe_codes']
        tokenizer.vocab = state['vocab']
        tokenizer.token2id = state['token2id']
        tokenizer.id2token = state['id2token']
        return tokenizer
