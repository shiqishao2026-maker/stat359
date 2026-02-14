import pickle
from gensim.models import Word2Vec
import gensim.downloader as api

# Load sentences from text8 corpus (same as in word2vec.py)
sentences = api.load("text8")
sent_list = list(sentences)

# Train a Word2Vec model using gensim
model = Word2Vec(
    sentences=sent_list,
    vector_size=100,  # match your PyTorch embedding dim if desired
    window=5,
    min_count=50,
    workers=4,
)

# Save the trained model in gensim's native format
model.save("word2vec_text8_gensim.model")

# Optionally, save the word vectors as a numpy array and word2idx mapping
word_vectors = model.wv.vectors
vocab = model.wv.index_to_key
word2idx = {word: idx for idx, word in enumerate(vocab)}

with open("word2vec_gensim_embeddings.pkl", "wb") as f:
    pickle.dump({
        'embeddings': word_vectors,
        'word2idx': word2idx,
        'idx2word': vocab
    }, f)

print(f"Trained gensim Word2Vec model on text8. Model and embeddings saved.")

