import numpy as np
import pandas as pd
from collections import Counter
import itertools
import pickle
import gensim.downloader as api
import nltk
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def main():
    # Load sentences from text8
    sentences = api.load("text8")
    sent_list = list(sentences)
    print(f"Number of sentences: {len(sent_list)}")
    print("First sample:", sent_list[0][:20])

    # Build word frequency counter
    counter = Counter()
    for sentence in sent_list:
        counter.update(sentence)
    print("Top 10 most common words:", counter.most_common(10))

    # Create word <-> index mapping
    word2idx = {word: i for i, (word, _) in enumerate(counter.items())}
    idx2word = {i: word for word, i in word2idx.items()}
    vocab_size = len(word2idx)
    print(f"Vocab size: {vocab_size}")

    # Skip-gram pair generator
    def skipgram_pairs(sentence, window_size):
        indices = [word2idx[w] for w in sentence if w in word2idx]
        pairs = []
        for i, center in enumerate(indices):
            for j in range(max(0, i - window_size), min(len(indices), i + window_size + 1)):
                if i != j:
                    pairs.append((center, indices[j]))
        return pairs

    # Generate skip-gram pairs for first 5 sentences
    window_size = 2
    skipgram_data = []
    for sentence in itertools.islice(sent_list, 5):
        skipgram_data.extend(skipgram_pairs(sentence, window_size))
    print(f"Number of skip-gram pairs (first 5 sentences): {len(skipgram_data)}")
    skipgram_df = pd.DataFrame(skipgram_data, columns=['center', 'context'])
    print("Sample skip-gram pairs:")
    print(skipgram_df.head())

    # Save processed data to pickle
    data_to_save = {
        'sent_list': sent_list,
        'counter': counter,
        'word2idx': word2idx,
        'idx2word': idx2word,
        'skipgram_df': skipgram_df
    }
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)
    print("Processed data saved to processed_data.pkl")

if __name__ == "__main__":
    main()


