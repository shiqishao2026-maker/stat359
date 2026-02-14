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
    full_counter = Counter()
    for sentence in sent_list:
        full_counter.update(sentence)
    print("Top 10 most common words:", full_counter.most_common(10))

    # only keep words that appear at least 50 times
    counter = Counter({k: c for k, c in full_counter.items() if c >= 50})
    trimmed_vocab = [word for word, count in counter.items()]
    
    # create mapping with reduced vocab
    word2idx = {word: i for i, word in enumerate(trimmed_vocab)}
    idx2word = {i: word for word, i in word2idx.items()}
    
    vocab_size = len(word2idx)
    print(f"Original Vocab Size: {len(counter)}")
    print(f"Trimmed Vocab Size (min_count=50): {vocab_size}")

    total_count = sum(counter.values())
    
    # Probability of keeping a word from original word2vec paper and implementation
    # downsample the frequent stop words to accelerate the training
    threshold = 1e-5 
    word_freqs = {w: c/total_count for w, c in counter.items()}
    p_keep = {w: (np.sqrt(f / threshold) + 1) * (threshold / f) for w, f in word_freqs.items()}
    
    # skipgram pair generator with subsampling
    def skipgram_pairs(sentence, window_size):
        np.random.seed(42)
        # convert to indices and apply subsampling
        indices = []
        for w in sentence:
            if w in word2idx:
                # only keep word if random number < probability of keeping
                if np.random.rand() < p_keep[w]:
                    indices.append(word2idx[w])
        
        # pair generation on the reduced sentence
        pairs = []
        for i, center in enumerate(indices):
            for j in range(max(0, i - window_size), min(len(indices), i + window_size + 1)):
                if i != j:
                    pairs.append((center, indices[j]))
        return pairs

    # Generate skip-gram pairs for all sentences
    window_size = 2
    skipgram_data = []
    for sentence in itertools.islice(sent_list, 1701):
        skipgram_data.extend(skipgram_pairs(sentence, window_size))
    print(f"Number of skip-gram pairs: {len(skipgram_data)}")
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

