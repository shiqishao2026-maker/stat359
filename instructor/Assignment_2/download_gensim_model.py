import os
import gensim.downloader as api
from gensim.models import KeyedVectors

# List of models to download
models = [
    "word2vec-google-news-300",
    "fasttext-wiki-news-subwords-300"
]
 

def load_local_gensim_model(model_name):
    """
    Load a gensim model from disk by name (without .model extension).
    If the model file does not exist, download and save it first.
    Example: model = load_local_gensim_model('word2vec-google-news-300')
    """
    import os
    from gensim.models import KeyedVectors
    import gensim.downloader as api
    path = f"{model_name}.model"
    if not os.path.exists(path):
        print(f"Model file {path} not found. Downloading '{model_name}'...")
        model = api.load(model_name)
        model.save(path)
        print(f"Model '{model_name}' downloaded and saved as '{path}'")
    print(f"Loading model from {path} ...")
    return KeyedVectors.load(path)
 
if __name__ == "__main__":
    for model_name in models:
        if not os.path.exists(f"{model_name}.model"):
            print(f"Model {model_name} not found locally. Downloading...")
            model = api.load(model_name)
            model.save(f"{model_name}.model")
            print(f"Model {model_name} downloaded and saved.")
        else:
            print(f"Model {model_name} already exists locally. skipping download.")