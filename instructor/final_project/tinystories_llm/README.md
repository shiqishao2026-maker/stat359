# TinyStories Transformer & Chat Model (hw4)

This folder contains code for training, using, and instruction-tuning a transformer-based language model on the TinyStories dataset, including a custom BPE tokenizer and chat interface.

## File Overview

### 1. `bpe_tokenizer.py`
Implements a Byte Pair Encoding (BPE) tokenizer from scratch, supporting special tokens for chat/instruction tuning. Includes methods for fitting on text, encoding/decoding, and saving/loading the tokenizer.

### 2. `train_bpe_tokenizer_hf.py`
Script to train the BPE tokenizer on the HuggingFace TinyStories dataset. Supports sampling a subset of data for faster training. Saves the tokenizer as a pickle file.

**Usage:**
```bash
python train_bpe_tokenizer_hf.py --sample 10000
```

### 3. `transformer_model.py`
Defines the transformer architecture for TinyStories, including configuration, embeddings, self-attention, feed-forward, and the full model for causal language modeling. Supports local attention windows and text generation (greedy, sampling, beam search).

### 4. `train_tinystories_model.py`
Trains a transformer language model on the TinyStories dataset using the custom BPE tokenizer. Supports configurable model size, training parameters, and AMP. Saves checkpoints and logs to TensorBoard.

**Usage:**
```bash
python train_tinystories_model.py --dataset roneneldan/TinyStories --output_dir tinystories_model
```

### 5. `train_tinystories_chat_model.py`
Instruction-tunes a pretrained TinyStories model for chat using a conversational dataset. Handles alternating user/assistant turns, special tokens, and evaluation. Supports resuming from checkpoints and AMP.

**Usage:**
```bash
python train_tinystories_chat_model.py --pretrained_model_path tinystories_model/best_model.pth --output_dir tinystories_chat_model
```

### 6. `generate_tinystories_text.py`
Generates text from a trained TinyStories model given a prompt. Supports temperature, top-k, and top-p sampling.

**Usage:**
```bash
python generate_tinystories_text.py --model_path tinystories_model/best_model.pth --prompt "Once upon a time,"
```

### 7. `chat_with_tinystories_model.py`
Interactive chat interface for a TinyStories chat model. Loads a trained model and tokenizer, and alternates between user and assistant turns in the console.

**Usage:**
```bash
python chat_with_tinystories_model.py --model_path tinystories_chat_model/final_model.pth
```

---

## Requirements

- Python 3.8+
- PyTorch
- HuggingFace Datasets
- tqdm
- numpy
- tensorboard

Install dependencies (if using Poetry):
```bash
poetry install
```

---

## Notes

- All models use a custom BPE tokenizer (`bpe_tokenizer_tinystories.pkl`).
- For chat/instruction tuning, special tokens like `<user>`, `<assistant>`, `<system>` are used.
- Training and generation scripts support CPU, CUDA, and Apple MPS devices.

---
