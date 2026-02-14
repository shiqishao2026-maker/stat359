import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    def __init__(self, data):
        self.centers = torch.as_tensor(data["center"].values, dtype=torch.long)
        self.contexts = torch.as_tensor(data["context"].values, dtype=torch.long)

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        return self.centers[idx], self.contexts[idx]
    

# Simple Skip-gram Module
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)
        bound = 0.5 / embedding_dim
        nn.init.uniform_(self.input_embeddings.weight, -bound, bound)
        nn.init.zeros_(self.output_embeddings.weight)

    def forward(self, center, context, negative):
        center_vec = self.input_embeddings(center)
        pos_context_vec = self.output_embeddings(context)
        pos_logits = (center_vec * pos_context_vec).sum(dim=-1)
        neg_context_vec = self.output_embeddings(negative)
        neg_logits = (center_vec.unsqueeze(1) * neg_context_vec).sum(dim=-1)
        return pos_logits, neg_logits
    
    def get_input_embeddings(self):
        return self.input_embeddings.weight
    
    def get_output_embeddings(self):
        return self.output_embeddings.weight

# Load processed data
with open("processed_data.pkl", "rb") as f:
    data = pickle.load(f)

skipgram_df = data["skipgram_df"]
counter = data["counter"]
word2idx = data["word2idx"]
idx2word = data["idx2word"]
vocab_size = len(word2idx)

# Precompute negative sampling distribution below
counts = torch.zeros(vocab_size, dtype=torch.float32)
for w, idx in word2idx.items():
    counts[idx] = float(counter.get(w, 0))

neg_dist = counts.pow(0.75)
neg_dist = neg_dist / neg_dist.sum() 

# Device selection: CUDA > MPS > CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")
neg_dist = neg_dist.to(device)

# Dataset and DataLoader
dataset = SkipGramDataset(skipgram_df)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, Loss, Optimizer
model = Word2Vec(vocab_size, EMBEDDING_DIM).to(device)
bce = nn.BCEWithLogitsLoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

@torch.no_grad()
def sample_negative_words(neg_dist, pos_context, K, max_rounds=10):
    B = pos_context.size(0)
    negatives = torch.multinomial(neg_dist, num_samples=B * K, replacement=True).view(B, K)
    repeated = negatives.eq(pos_context.unsqueeze(1))
    rounds = 0
    while repeated.any() and rounds < max_rounds:
        resample = torch.multinomial(neg_dist, num_samples=repeated.sum().item(), replacement=True)
        negatives[repeated] = resample
        repeated = negatives.eq(pos_context.unsqueeze(1))
        rounds += 1

    if repeated.any():
        negatives[repeated] = (negatives[repeated] + 1) % neg_dist.numel()
    
    return negatives

def make_targets(center, context, vocab_size):
    return None

# Training loop
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    n_batches = 0
    for centers, contexts in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        centers = centers.to(device)
        contexts = contexts.to(device)
        negatives = sample_negative_words(neg_dist, contexts, NEGATIVE_SAMPLES)

        optimizer.zero_grad()
        pos_logits, neg_logits = model(centers, contexts, negatives)

        pos_labels = torch.ones_like(pos_logits, device=device)
        neg_labels = torch.zeros_like(neg_logits, device=device)

        loss_pos = bce(pos_logits, pos_labels)
        loss_neg = bce(neg_logits, neg_labels)
        loss = loss_pos + loss_neg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    print(f"Epoch {epoch+1}: avg_loss = {total_loss / max(n_batches, 1):.6f}")

# Save embeddings and mappings

embeddings = model.get_input_embeddings().detach().cpu().numpy()

with open('word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'word2idx': data['word2idx'], 'idx2word': data['idx2word']}, f)
print("Embeddings saved to word2vec_embeddings.pkl")
