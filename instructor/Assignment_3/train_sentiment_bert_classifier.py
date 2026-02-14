#!/usr/bin/env python
# coding: utf-8
"""
Script for training a BERT-based sentiment classifier on the financial_phrasebank dataset.
"""

# ========== Imports ==========
import numpy as np
import pandas as pd
import datasets
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
from transformers import BertTokenizer, BertModel

print("\n========== Loading Dataset ==========")
# ========== Load Dataset ==========
dataset = datasets.load_dataset('financial_phrasebank', 'sentences_50agree', trust_remote_code=True)
print("Dataset loaded. Example:", dataset['train'][0])

print("\n========== Preparing DataFrame ==========")
data = pd.DataFrame(dataset['train'])
data['text_label'] = data['label'].apply(lambda x: 'positive' if x == 2 else 'neutral' if x == 1 else 'negative')
print(f"DataFrame shape: {data.shape}")

# ========== Tokenization ==========
print("\n========== Loading BERT Tokenizer & Model ==========")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

max_seq_len = 32

def get_device():
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def encode_sentences(sentences, tokenizer, max_len=32):
    encodings = tokenizer(list(sentences), truncation=True, padding='max_length', max_length=max_len, return_tensors='pt')
    return encodings

print("\n========== Encoding Sentences ==========")
encodings = encode_sentences(data['sentence'], tokenizer, max_len=max_seq_len)
y = data['label'].values
print(f"Input IDs shape: {encodings['input_ids'].shape}, y shape: {y.shape}")

# ========== Train/Test Split ==========
print("\n========== Splitting Data ==========")
X_trainval, X_test, y_trainval, y_test = train_test_split(
    np.arange(len(y)), y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=42
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ========== PyTorch Dataset ==========
class BertDataset(Dataset):
    def __init__(self, encodings, labels, indices):
        self.input_ids = encodings['input_ids'][indices]
        self.attention_mask = encodings['attention_mask'][indices]
        self.labels = torch.tensor(labels[indices], dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

train_dataset = BertDataset(encodings, y, X_train)
val_dataset = BertDataset(encodings, y, X_val)
test_dataset = BertDataset(encodings, y, X_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print("DataLoaders created.")

# ========== Model Definition ==========
print("\n========== Defining BERT Classifier ==========")
class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_classes, dropout=0.3):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        return self.fc(x)

num_classes = len(np.unique(y))
model = BertClassifier(bert_model, num_classes)

# ========== Training Setup ==========
print("\n========== Setting Up Training ==========")
device = get_device()
print(f"Using device: {device}")
os.makedirs("outputs", exist_ok=True)
model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
counts = [684, 2879, 1363]  # Class counts
class_weights = 1. / torch.tensor(counts, dtype=torch.float)
class_weights = class_weights / class_weights.sum()
class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
print("Training setup complete.")

# ========== Training Loop ==========
print("\n========== Starting Training Loop ==========")
num_epochs = 5
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
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * input_ids.size(0)
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
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * input_ids.size(0)
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
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), 'outputs/best_bert_model.pth')
        print(f'>>> Saved new best model (Val F1: {best_val_f1:.4f})')

# ========== Plot Learning Curves ==========
print("\n========== Plotting Learning Curves ==========")
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
plt.savefig('outputs/bert_f1_learning_curves.png')
plt.show()
print("Learning curves saved as 'outputs/bert_f1_learning_curves.png'.")

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
plt.savefig('outputs/bert_accuracy_learning_curve.png')
plt.show()
print("Accuracy curve saved as 'outputs/bert_accuracy_learning_curve.png'.")

# ========== Test Evaluation ==========
print("\n========== Evaluating on Test Set ==========")
model.load_state_dict(torch.load('outputs/best_bert_model.pth'))
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask)
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
plt.savefig('outputs/bert_confusion_matrix.png')
plt.show()
print("Confusion matrix saved as 'outputs/bert_confusion_matrix.png'.")
print("\nPer-class F1 Scores:")
for i, name in enumerate(class_names):
    class_f1 = f1_score(all_labels, all_preds, labels=[i], average='macro')
    print(f"{name}: {class_f1:.4f}")

print("\n========== Script Complete ==========")
# End of script
