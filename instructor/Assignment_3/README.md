# Homework 3: Sentiment Classification with Neural Networks

This directory contains code, models, and results for sentiment classification experiments using various neural network architectures, including MLP, RNN, LSTM, GRU, BERT, and GPT. The goal is to train and evaluate different models on sentiment analysis tasks and compare their performance.

## Contents

- **train_sentiment_rnn_classifier.py**: Train a sentiment classifier using a basic RNN architecture.
- **train_sentiment_gru_classifier.py**: Train a sentiment classifier using a GRU architecture.
- **train_sentiment_bert_classifier.py**: Train a sentiment classifier using a BERT-based model.
- **train_sentiment_gpt_classifier.py**: Train a sentiment classifier using a GPT-based model.

### Saved Model Files
- **best_*.pth**: Saved PyTorch model weights for the best-performing models (RNN, GRU, BERT, GPT).

### Evaluation Results
- **accuracy_learning_curve.png**: Accuracy learning curve for the models.
- **f1_learning_curves.png**: F1 score learning curves for the models.
- **confusion_matrix.png**: Confusion matrix for the best model.
- **[model]_accuracy_learning_curve.png**: Accuracy learning curve for the specific model.
- **[model]_f1_learning_curves.png**: F1 score learning curves for the specific model.
- **[model]_confusion_matrix.png**: Confusion matrix for the specific model.

## Usage

1. **Training**: Run the desired training script (e.g., `python train_sentiment_lstm_classifier.py`) to train a model. Each script will save the best model weights and output evaluation metrics.
2. **Evaluation**: After training, use the generated plots and confusion matrices to analyze model performance.

## Requirements

- Python 3.x
- PyTorch
- Transformers (for BERT/GPT models)
- Other dependencies as specified in the project root (see `pyproject.toml` or `env.yml`)

## Notes

- Each script is self-contained and can be run independently.
- The saved model files (`.pth`) can be loaded for further evaluation or inference.
- Plots and results are generated automatically after training.

---

For more details, refer to the code and comments within each script.
