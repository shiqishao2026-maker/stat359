import pickle
import numpy as np

# Load embeddings and mappings
with open('word2vec_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)
embeddings = data['embeddings']
word2idx = data['word2idx']
idx2word = data['idx2word']

# Example: get embedding for a word
def get_vector(word):
    idx = word2idx.get(word)
    if idx is None:
        print(f"Word '{word}' not in vocabulary.")
        return None
    return embeddings[idx]

# Example usage
word = 'huge'
vec = get_vector(word)
if vec is not None:
    print(f"Embedding for '{word}':\n", vec)

# Find most similar word (cosine similarity)
def most_similar(query_word, topn=5, metric="cosine"):
    qvec = get_vector(query_word)
    if qvec is None:
        return []
    if metric == "cosine":
        sims = embeddings @ qvec / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(qvec) + 1e-9)
        best = np.argsort(-sims)
        return [(idx2word[i], float(sims[i])) for i in best[1:topn+1]]
    elif metric == "euclidean":
        dists = np.linalg.norm(embeddings - qvec, axis=1)
        best = np.argsort(dists)
        return [(idx2word[i], float(dists[i])) for i in best[1:topn+1]]
    elif metric == "dot":
        sims = embeddings @ qvec
        best = np.argsort(-sims)
        return [(idx2word[i], float(sims[i])) for i in best[1:topn+1]]
    else:
        raise ValueError(f"Unknown metric: {metric}")

# Example usage
print(f"Most similar to '{word}' (euclidean):", most_similar(word, metric="euclidean"))
print(f"Most similar to '{word}' (cosine):", most_similar(word, metric="cosine"))
print(f"Most similar to '{word}' (dot):", most_similar(word, metric="dot"))
