from datasets import load_dataset
from bpe_tokenizer import BPETokenizer
import argparse

# Parameters
data_name = "roneneldan/TinyStories"
split = "train"
vocab_size = 10000
save_path = "bpe_tokenizer_tinystories.pkl"

# Load dataset in streaming mode
dataset = load_dataset(data_name, split=split, streaming=True)

parser = argparse.ArgumentParser(description="Train BPE tokenizer with optional data sampling.")
parser.add_argument('--sample', type=int, default=0, help='Number of samples to use (<=0 for all data)')
args = parser.parse_args()

def text_generator():
    count = 0
    for item in dataset:
        if "text" in item:
            yield item["text"]
            count += 1
            if args.sample > 0 and count >= args.sample:
                break

# Define special tokens for GPT/instruction tuning
special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>', '<user>', '<assistant>', '<system>']

# Train tokenizer
tokenizer = BPETokenizer(vocab_size=vocab_size, special_tokens=special_tokens)
tokenizer.fit(text_generator())

# Save tokenizer
tokenizer.save(save_path)
print(f"Trained BPE tokenizer saved to {save_path}")
