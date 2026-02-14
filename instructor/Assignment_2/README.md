# Homework 2: Word Embeddings and Visualization

This folder contains code and resources for exploring, training, and visualizing word embeddings as part of the LLM 2025 Summer course.

## Contents

- `data.py` - Data processing utilities.
- `download_gensim_model.py` - Script to download pre-trained Gensim models.
- `gensim_train_word2vec.py` - Script to train Word2Vec models using Gensim.
- `pytorch_train_word2vec.py` - Script to train Word2Vec models using PyTorch.
- `pytorch_show_embeddings.py` - Visualization of embeddings trained with PyTorch.
- `inspect_embeddings.ipynb` - Jupyter notebook for embedding visualization and analysis.
- Pre-trained models and data files:
  - `fasttext-wiki-news-subwords-300.model` and `.npy`
  - `word2vec-google-news-300.model` and `.npy`
  - `word2vec_text8_gensim.model`
  - `word2vec_embeddings.pkl`, `word2vec_gensim_embeddings.pkl`
  - `processed_data.pkl`

## Run Dependencies

The following Python packages are required to run the scripts in this folder:

- Python >= 3.11
- numpy
- pandas
- gensim
- torch
- scikit-learn
- matplotlib
- ipywidgets
- nltk
- tqdm
- datasets

You can install these dependencies using Poetry (see `pyproject.toml` in the project root). Some scripts and notebooks may also require Jupyter.

## How to Run Scripts

To run any script, use:

```
python script_name.py
```

If you set up the environment using the course-provided config files and Poetry:

- After activating the environment (`poetry shell`), use:
  ```
  python script_name.py
  ```

- Otherwise, use:

```
poetry run python script_name.py
```

### Required sequence to run each step:


- Download data to train models
  ```
  python data.py
  ```

- Download pre-trained Gensim models (required for embedding comparison in `inspect_embeddings.ipynb`):
  ```
  python download_gensim_model.py
  ```

- Train a Word2Vec model using Gensim:
  ```
  python gensim_train_word2vec.py
  ```

- Train a Word2Vec model using PyTorch:
  ```
  python pytorch_train_word2vec.py
  ```
- Test embeddings trained with PyTorch:
  ```
  python pytorch_show_embeddings.py
  ```

- Inspect embeddings:
  `inspect_embeddings.ipynb`

- Render the notebook to HTML for submission:
  ```
  quarto render inspect_embeddings.ipynb --to html
  ```


