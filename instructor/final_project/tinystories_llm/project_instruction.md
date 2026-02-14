# Table of Contents
- [Table of Contents](#table-of-contents)
- [Background and Motivation](#background-and-motivation)
- [Course Project: Training a Decoder-Only (GPT-Style) Model with TinyStories](#course-project-training-a-decoder-only-gpt-style-model-with-tinystories)
  - [Part 1: Training Experience](#part-1-training-experience)
    - [Prerequisites](#prerequisites)
    - [Environment Setup](#environment-setup)
    - [Step 1: Train the Tokenizer](#step-1-train-the-tokenizer)
    - [Step 2: Train the Foundation Model](#step-2-train-the-foundation-model)
    - [Step 3: Test Your Model](#step-3-test-your-model)
    - [Step 4: Instruction Tuning](#step-4-instruction-tuning)
    - [Step 5: Chat with Your Model](#step-5-chat-with-your-model)

# Background and Motivation

Recent advances in language modeling have shown that large-scale models, such as GPT-3 and GPT-4, are capable of generating coherent and creative text. However, smaller language models (SLMs) with fewer parameters often struggle to produce fluent and consistent English, raising questions about the minimal requirements for language understanding and generation.

The TinyStories project, introduced by Eldan and Li (2023), addresses this by creating a synthetic dataset of short stories using only vocabulary that a typical 3-4 year-old child would understand. This dataset enables the training and evaluation of much smaller models (with fewer than 10 million parameters) that can still generate diverse, grammatically correct, and contextually consistent stories. The project also proposes a novel evaluation framework using GPT-4 to grade generated stories on grammar, creativity, and consistency, providing a more nuanced assessment than traditional benchmarks.

**Reference:** [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759) by Ronen Eldan and Yuanzhi Li (2023).

**In this project, you will replicate the training of models as described in the TinyStories paper.** You will train a decoder-only (GPT-style) model from scratch on the TinyStories dataset, analyze its capabilities, and explore instruction tuning for conversational tasks. This hands-on experience will help you understand the challenges and opportunities in training small language models and provide insights into the emergence of language abilities at smaller scales.

---

# Course Project: Training a Decoder-Only (GPT-Style) Model with TinyStories

In this course project, you will train a decoder-only (GPT-style) language model from scratch using the TinyStories dataset. You will also learn how to prepare a conversational dataset for instruction tuning, enabling your model to follow a chat-style interface similar to ChatGPT.

---

## Part 1: Training Experience

In this part, you will go through the complete process of training a language model. All necessary code has been provided; your main task is to follow the process and create a small, yet non-trivial, model.

### Prerequisites

Training a foundation model is a non-trivial task. Even though we have restricted the model size and dataset to be small ("tiny"), you will still need access to a GPU instance. Please follow the previous instructions to set up a Google Cloud Platform (GCP) instance. A T4 or L4 GPU instance is recommended (L4 is more powerful).

### Environment Setup

1. **Clone the course repository** to your GPU instance.
2. **Install dependencies** using Poetry:
   ```bash
   poetry install
   ```
   This will create a virtual environment and install all required packages for the project.

### Step 1: Train the Tokenizer

Use the provided script to train a tokenizer:
```bash
python train_bpe_tokenizer_hf.py
```

**What this script does:**
- Loads the TinyStories dataset (`roneneldan/TinyStories`) in streaming mode, so it doesn't require downloading the entire dataset at once.
- Optionally allows you to specify how many samples to use with the `--sample` argument (e.g., `--sample 10000` to use only 10,000 stories for faster testing).
- Iterates through the dataset and extracts the `"text"` field from each story.
- Defines special tokens needed for GPT-style and instruction-tuned models (e.g., `<pad>`, `<unk>`, `<bos>`, `<eos>`, `<user>`, `<assistant>`, `<system>`).
- Trains a Byte-Pair Encoding (BPE) tokenizer with a vocabulary size of 10,000.
- Saves the trained tokenizer to `bpe_tokenizer_tinystories.pkl`.

**Output:**
- The trained tokenizer will be saved as `bpe_tokenizer_tinystories.pkl` in the current directory.

### Step 2: Train the Foundation Model

Train your model using:
```bash
python train_tinystories_model.py
```
- Save the output logs for later analysis.
- You may adjust training parameters for faster testing or better results.
- Use the `--amp` option to enable half-precision (float16) training for improved speed:
  ```bash
  python train_tinystories_model.py --amp
  ```

**What this script does:**
- The script loads the BPE tokenizer (`bpe_tokenizer_tinystories.pkl`) and the TinyStories dataset from HuggingFace.
- It prepares the dataset for training and validation, tokenizing and padding each story to a fixed length.
- Builds a decoder-only transformer model (GPT-style) with configurable parameters (number of layers, hidden size, attention heads, etc.).
- Supports resuming from checkpoints, gradient accumulation, mixed-precision training (`--amp`), and logging with TensorBoard.
- Trains the model using cross-entropy loss, periodically evaluates on the validation set, and saves the best model and checkpoints.
- At the end of training, saves the final model and generates a sample story for qualitative evaluation.

**Key arguments you can use:**
- `--batch_size`: Set the batch size (default: 64)
- `--epochs`: Number of training epochs (default: 5)
- `--lr`: Learning rate (default: 1e-3)
- `--max_seq_len`: Maximum sequence length (default: 256)
- `--output_dir`: Directory to save models and logs
- `--resume_from_checkpoint`: Resume training from a saved checkpoint
- `--pilot_run`: Use a small subset of data for a quick test run

**Output:**
- Model checkpoints and logs will be saved in the specified output directory (default: `tinystories_model`).
- The best model (lowest validation loss) is saved as `best_model.pth`.
- The final model is saved as `final_model.pth`.
- Training and validation loss/perplexity are logged for analysis.

### Step 3: Test Your Model

Generate text with your trained model:
```bash
python generate_tinystories_text.py
```
- Evaluate whether the generated text is meaningful.

### Step 4: Instruction Tuning

Fine-tune your model for conversational abilities:
```bash
python train_tinystories_chat_model.py --pretrained_model_path <path_to_foundation_model>
```

**What this script does:**
- Loads the BPE tokenizer (`bpe_tokenizer_tinystories.pkl`) and a conversational TinyStories dataset from HuggingFace (`tinystories-conversations` by default).
- Prepares the dataset for instruction tuning: Each conversation is formatted as alternating `<user>` and `<assistant>` turns, tokenized, and padded/truncated to a fixed length.
- Loads your pretrained foundation model weights (from Step 2) and adapts the model for chat-style instruction tuning.
- Uses cross-entropy loss, gradient accumulation, mixed-precision training (`--amp`), and logs with TensorBoard.
- Periodically evaluates on a validation set, saves the best model, and checkpoints training progress.
- At the end, saves the final instruction-tuned model.

**Key arguments you can use:**
- `--pretrained_model_path`: Path to your foundation model checkpoint (required)
- `--batch_size`, `--epochs`, `--lr`, `--max_seq_len`, etc. (see script for full list)
- `--pilot_run`: Use a small subset of data for a quick test run

**Output:**
- Instruction-tuned model checkpoints and logs are saved in the specified output directory (default: `tinystories_chat_model`).
- The best model is saved as `best_model.pth`, and the final model as `final_model.pth`.

### Step 5: Chat with Your Model

Interact with your instruction-tuned model:
```bash
python chat_with_tinystories_model.py
```

