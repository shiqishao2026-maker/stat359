#!/usr/bin/env python
# coding: utf-8
"""
Script for training an RNN-based sentiment classifier on the financial_phrasebank dataset.
"""

# ========== Imports ==========
import numpy as np
import pandas as pd
import datasets
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

print("\n========== Loading Dataset ==========")
# ========== Load Dataset ==========
dataset = datasets.load_dataset('financial_phrasebank', 'sentences_50agree', trust_remote_code=True)
print("Dataset loaded. Example:", dataset['train'][0])

print("\n========== Preparing DataFrame ==========")
data = pd.DataFrame(dataset['train'])
data['text_label'] = data['label'].apply(lambda x: 'positive' if x == 2 else 'neutral' if x == 1 else 'negative')
print(f"DataFrame shape: {data.shape}")

# Print distribution of sentence lengths
sentence_lengths = data['sentence'].apply(lambda x: len(x.split()))
print("\nSentence length statistics:")
print(sentence_lengths.describe())
plt.figure(figsize=(10,6))
plt.hist(sentence_lengths, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Sentence Lengths')
plt.xlabel('Sentence Length (words)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# ========== Tokenization and Embedding ==========
print("\n========== Loading SentenceTransformer Model ==========")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

def get_device():
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def tokenize_and_pad(sentences, model, max_len=32):
    device = get_device()
    model = model.to(device)
    emb_dim = model.get_sentence_embedding_dimension()
    embeddings = []
    for sent in tqdm(sentences, desc="Tokenizing & encoding"):
        tokens = model.tokenizer(sent, truncation=True, padding='max_length', max_length=max_len, return_tensors='pt')
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)
        with torch.no_grad():
            output = model._first_module().auto_model(input_ids=input_ids, attention_mask=attention_mask)
            emb = output.last_hidden_state.squeeze(0)
            if emb.shape[0] < max_len:
                pad = torch.zeros(max_len - emb.shape[0], emb_dim, device=device)
                emb = torch.cat([emb, pad], dim=0)
            elif emb.shape[0] > max_len:
                emb = emb[:max_len, :]
            embeddings.append(emb.cpu().numpy())
    return np.stack(embeddings)

print("\n========== Encoding Sentences as Sequences ==========")
max_seq_len = 32
X_seq = tokenize_and_pad(data['sentence'], embedding_model, max_len=max_seq_len)
y = data['label'].values
print(f"X_seq shape: {X_seq.shape}, y shape: {y.shape}")

# ========== Train/Test Split ==========
print("\n========== Splitting Data ==========")
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_seq, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=42
)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# ========== PyTorch Dataset ==========
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = SequenceDataset(X_train, y_train)
val_dataset = SequenceDataset(X_val, y_val)
test_dataset = SequenceDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print("DataLoaders created.")

# ========== Model Definition ==========
print("\n========== Defining RNN Model ==========")
class RNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.5):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, nonlinearity='tanh', dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, h_n = self.rnn(x)  # h_n: (num_layers, batch, hidden_dim)
        out = h_n[-1]  # (batch, hidden_dim)
        return self.fc(out)

input_dim = X_train.shape[2]
num_classes = len(np.unique(y))
hidden_dim = 128
num_layers = 2
model = RNNClassifier(input_dim, hidden_dim, num_layers, num_classes)
print(f"Model initialized with input_dim={input_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}, num_classes={num_classes}")

# ========== Training Setup ==========
print("\n========== Setting Up Training ==========")
device = get_device()
print(f"Using device: {device}")
os.makedirs("outputs", exist_ok=True)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
counts = [684, 2879, 1363]  # Class counts
class_weights = 1. / torch.tensor(counts, dtype=torch.float)
class_weights = class_weights / class_weights.sum()
class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
print("Training setup complete.")

# ========== Training Loop ==========
print("\n========== Starting Training Loop ==========")
num_epochs = 30
best_val_f1 = 0.0
train_loss_history = []
val_loss_history = []
train_f1_history = []
val_f1_history = []
train_acc_history = []
val_acc_history = []

for epoch in range(num_epochs):
    print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
    model.train()
    running_loss = 0.0
    all_train_preds = []
    all_train_labels = []
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_train_preds.extend(preds.cpu().numpy())
        all_train_labels.extend(labels.cpu().numpy())
    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')
    train_acc = (np.array(all_train_preds) == np.array(all_train_labels)).mean()
    train_loss_history.append(epoch_train_loss)
    train_f1_history.append(train_f1)
    train_acc_history.append(train_acc)
    print(f"Train Loss: {epoch_train_loss:.4f}, Train F1: {train_f1:.4f}, Train Acc: {train_acc:.4f}")
    # Validation
    model.eval()
    val_loss = 0.0
    all_val_preds = []
    all_val_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())
    epoch_val_loss = val_loss / len(val_loader.dataset)
    val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
    val_acc = (np.array(all_val_preds) == np.array(all_val_labels)).mean()
    val_loss_history.append(epoch_val_loss)
    val_f1_history.append(val_f1)
    val_acc_history.append(val_acc)
    print(f"Val Loss: {epoch_val_loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}")
    scheduler.step(val_f1)
    print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {epoch_train_loss:.4f} | Train F1: {train_f1:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}')
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), 'outputs/best_rnn_model.pth')
        print(f'>>> Saved new best model (Val F1: {best_val_f1:.4f})')

# ========== Plot Learning Curves ==========
print("\n========== Plotting Learning Curves ==========")
os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(12, 15))
plt.subplot(3, 1, 1)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.subplot(3, 1, 2)
plt.plot(train_f1_history, label='Train F1')
plt.plot(val_f1_history, label='Val F1')
plt.title('F1 Macro Score Curve')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True)
plt.subplot(3, 1, 3)
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(val_acc_history, label='Val Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/rnn_f1_learning_curves.png')
plt.show()
print("Learning curves saved as 'outputs/rnn_f1_learning_curves.png'.")

# Save accuracy plot separately
plt.figure(figsize=(8, 6))
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(val_acc_history, label='Val Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/rnn_accuracy_learning_curve.png')
plt.show()
print("Accuracy curve saved as 'outputs/rnn_accuracy_learning_curve.png'.")

# ========== Test Evaluation ==========
print("\n========== Evaluating on Test Set ==========")
model.load_state_dict(torch.load('outputs/best_rnn_model.pth'))
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
test_acc = (np.array(all_preds) == np.array(all_labels)).mean()
test_f1_macro = f1_score(all_labels, all_preds, average='macro')
test_f1_weighted = f1_score(all_labels, all_preds, average='weighted')
print('\n' + '='*50)
print(f"Final Test Accuracy: {test_acc:.4f}")
print(f"Test F1 Macro: {test_f1_macro:.4f}")
print(f"Test F1 Weighted: {test_f1_weighted:.4f}")
print('='*50 + '\n')
class_names = ['Negative (0)', 'Neutral (1)', 'Positive (2)']
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.savefig('outputs/rnn_confusion_matrix.png')
plt.show()
print("Confusion matrix saved as 'outputs/rnn_confusion_matrix.png'.")
print("\nPer-class F1 Scores:")
for i, name in enumerate(class_names):
    class_f1 = f1_score(all_labels, all_preds, labels=[i], average='macro')
    print(f"{name}: {class_f1:.4f}")

print("\n========== Script Complete ==========")
# End of script
