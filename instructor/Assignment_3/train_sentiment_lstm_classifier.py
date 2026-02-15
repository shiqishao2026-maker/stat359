import os
import json
import random
import argparse
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import gensim.downloader as api
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tokenize(text):
    return _TOKEN_RE.findall(text.lower())


def stratified_splits(texts, labels, seed, test_size=0.15, val_size_within_trainval=0.15):
    X = np.array(texts, dtype=object)
    y = np.array(labels, dtype=np.int64)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size_within_trainval,
        random_state=seed,
        stratify=y_trainval,
    )
    return (X_train.tolist(), y_train.tolist()), (X_val.tolist(), y_val.tolist()), (X_test.tolist(), y_test.tolist())


def compute_class_weights(y_train, num_classes):
    counts = np.bincount(np.array(y_train, dtype=np.int64), minlength=num_classes).astype(np.float64)
    total = counts.sum()
    weights = total / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def build_padded_sequences(ft, texts, seq_len=32, dim=300):
    cache = {}
    X = np.zeros((len(texts), seq_len, dim), dtype=np.float32)

    for i, s in enumerate(texts):
        toks = tokenize(s)[:seq_len]
        for j, tok in enumerate(toks):
            if tok in cache:
                v = cache[tok]
            else:
                if tok in ft.key_to_index:
                    v = ft.get_vector(tok).astype(np.float32)
                else:
                    v = np.zeros((dim,), dtype=np.float32)
                cache[tok] = v
            X[i, j] = v

    return X


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(np.array(y, dtype=np.int64), dtype=torch.long)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, bidirectional, num_classes):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        out_dim = hidden_dim * self.num_directions
        mid = max(out_dim // 2, 32)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, num_classes),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            h_fwd = h_n[-2]
            h_bwd = h_n[-1]
            h = torch.cat([h_fwd, h_bwd], dim=1)
        else:
            h = h_n[-1]

        return self.head(h)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    all_y = []
    all_pred = []
    total_loss = 0.0
    n = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = loss_fn(logits, yb)

        bs = yb.size(0)
        total_loss += loss.item() * bs
        n += bs

        pred = torch.argmax(logits, dim=1)
        all_y.append(yb.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    avg_loss = total_loss / max(n, 1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return avg_loss, acc, f1, y_true, y_pred


def plot_curves(history, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    epochs = history["epoch"]

    def plot_one(k_tr, k_va, ylabel, filename):
        plt.figure()
        plt.plot(epochs, history[k_tr], label="train")
        plt.plot(epochs, history[k_va], label="val")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), dpi=200)
        plt.close()

    plot_one("train_loss", "val_loss", "Loss", f"{prefix}_loss_curve.png")
    plot_one("train_acc", "val_acc", "Accuracy", f"{prefix}_acc_curve.png")
    plot_one("train_f1", "val_f1", "Macro F1", f"{prefix}_f1_curve.png")


def plot_confusion(y_true, y_pred, label_names, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    disp.plot(ax=ax, values_format="d")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--min_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="outputs_lstm")
    args = parser.parse_args()

    cfg = {}
    cfg["seed"] = args.seed
    cfg["batch_size"] = args.batch_size
    cfg["epochs"] = args.epochs
    cfg["min_epochs"] = args.min_epochs
    cfg["patience"] = args.patience
    cfg["lr"] = args.lr
    cfg["weight_decay"] = args.weight_decay
    cfg["hidden_dim"] = args.hidden_dim
    cfg["num_layers"] = args.num_layers
    cfg["dropout"] = args.dropout
    cfg["bidirectional"] = True if args.bidirectional else False
    cfg["seq_len"] = args.seq_len
    cfg["grad_clip"] = args.grad_clip
    cfg["output_dir"] = args.output_dir

    os.makedirs(cfg["output_dir"], exist_ok=True)
    set_seed(cfg["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    print("\n========== Loading Dataset ==========")
    ds = load_dataset("financial_phrasebank", "sentences_50agree", trust_remote_code=True)
    texts = ds["train"]["sentence"]
    labels = ds["train"]["label"]
    label_names = ds["train"].features["label"].names
    num_classes = len(label_names)

    (tr_texts, tr_y), (va_texts, va_y), (te_texts, te_y) = stratified_splits(
        texts, labels, seed=cfg["seed"], test_size=0.15, val_size_within_trainval=0.15
    )

    print("Label names:", label_names)
    print(f"Split sizes: train={len(tr_texts)}, val={len(va_texts)}, test={len(te_texts)}")
    print(f"Using seq_len={cfg['seq_len']} (pad/truncate)")
    print("Bidirectional:", cfg["bidirectional"])

    print("\n========== Loading FastText (Gensim) ==========")
    ft = api.load("fasttext-wiki-news-subwords-300")

    print("\n========== Building padded FastText sequences ==========")
    X_train = build_padded_sequences(ft, tr_texts, seq_len=cfg["seq_len"], dim=300)
    X_val = build_padded_sequences(ft, va_texts, seq_len=cfg["seq_len"], dim=300)
    X_test = build_padded_sequences(ft, te_texts, seq_len=cfg["seq_len"], dim=300)

    train_ds = SeqDataset(X_train, tr_y)
    val_ds = SeqDataset(X_val, va_y)
    test_ds = SeqDataset(X_test, te_y)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)

    class_w = compute_class_weights(tr_y, num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)

    model = LSTMClassifier(
        input_dim=300,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        bidirectional=cfg["bidirectional"],
        num_classes=num_classes,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=2, threshold=1e-4
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    best_val_f1 = -1.0
    best_path = os.path.join(cfg["output_dir"], "best_lstm.pt")
    bad_epochs = 0

    print("\n========== Training (LSTM) ==========")
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        all_y = []
        all_pred = []
        n = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["grad_clip"])
            opt.step()

            bs = yb.size(0)
            total_loss += loss.item() * bs
            n += bs

            pred = torch.argmax(logits, dim=1)
            all_y.append(yb.detach().cpu().numpy())
            all_pred.append(pred.detach().cpu().numpy())

        y_true = np.concatenate(all_y)
        y_pred = np.concatenate(all_pred)
        tr_loss = total_loss / max(n, 1)
        tr_acc = accuracy_score(y_true, y_pred)
        tr_f1 = f1_score(y_true, y_pred, average="macro")

        va_loss, va_acc, va_f1, _, _ = evaluate(model, val_loader, loss_fn, device)

        prev_lr = opt.param_groups[0]["lr"]
        scheduler.step(va_f1)
        new_lr = opt.param_groups[0]["lr"]
        if new_lr < prev_lr:
            print(f"LR reduced: {prev_lr:.2e} -> {new_lr:.2e}")

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)

        lr_now = opt.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | lr={lr_now:.2e} | "
            f"train loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f} | "
            f"val loss={va_loss:.4f} acc={va_acc:.4f} f1={va_f1:.4f}"
        )

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            bad_epochs = 0
            torch.save({"model_state": model.state_dict(), "config": cfg, "label_names": label_names}, best_path)
        else:
            bad_epochs += 1

        if epoch >= cfg["min_epochs"] and bad_epochs >= cfg["patience"]:
            print(f"Early stopping at epoch {epoch} (no val F1 improvement for {cfg['patience']} epochs).")
            break

    plot_curves(history, cfg["output_dir"], prefix="lstm")
    with open(os.path.join(cfg["output_dir"], "history_lstm.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    te_loss, te_acc, te_f1, te_y_true, te_y_pred = evaluate(model, test_loader, loss_fn, device)
    print("\n========== Test (LSTM) ==========")
    print(f"Test loss={te_loss:.4f} acc={te_acc:.4f} macro_f1={te_f1:.4f}")

    cm_path = os.path.join(cfg["output_dir"], "lstm_confusion_matrix.png")
    plot_confusion(te_y_true, te_y_pred, label_names, cm_path)

    with open(os.path.join(cfg["output_dir"], "test_metrics_lstm.json"), "w", encoding="utf-8") as f:
        json.dump({"test_loss": te_loss, "test_acc": te_acc, "test_macro_f1": te_f1}, f, indent=2)

    print(f"\nSaved: {best_path}")
    print(f"Saved curves: {cfg['output_dir']}/lstm_*_curve.png")
    print(f"Saved confusion matrix: {cm_path}")


if __name__ == "__main__":
    main()
