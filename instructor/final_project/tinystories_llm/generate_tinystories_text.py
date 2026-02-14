import torch
import argparse
from bpe_tokenizer import BPETokenizer
from transformer_model import TinyStoriesConfig, TinyStoriesForCausalLM
import os


def load_tokenizer(tokenizer_path):
    return BPETokenizer.load(tokenizer_path)


def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained TinyStories model.")
    parser.add_argument('--model_path', type=str, default='tinystories_model/best_model.pth', help='Path to the trained model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, default='bpe_tokenizer_tinystories.pkl', help='Path to the BPE tokenizer')
    parser.add_argument('--prompt', type=str, default='Once upon a time, there was a', help='Prompt to start text generation')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=0, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (nucleus) sampling')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], help='Device to use')
    args = parser.parse_args()

    # Device selection
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer_path)

    # Load model config
    config_path = os.path.join(os.path.dirname(args.model_path), 'args.json')
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            train_args = json.load(f)
        config = TinyStoriesConfig(
            vocab_size=len(tokenizer.token2id),
            hidden_size=train_args.get('hidden_size', 256),
            num_hidden_layers=train_args.get('num_layers', 4),
            num_attention_heads=train_args.get('num_heads', 8),
            intermediate_size=train_args.get('intermediate_size', 1024),
            hidden_dropout_prob=train_args.get('dropout', 0.1),
            attention_probs_dropout_prob=train_args.get('dropout', 0.1),
            max_position_embeddings=train_args.get('max_seq_len', 512),
            window_size=train_args.get('window_size', 256),
        )
    else:
        config = TinyStoriesConfig(vocab_size=len(tokenizer.token2id))

    # Load model
    model = TinyStoriesForCausalLM(config)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(args.prompt, add_special_tokens=True)], dtype=torch.long).to(device)
    eos_token_id = tokenizer.token2id.get('<eos>', None)
    print("eos_token_id:", eos_token_id)   
    # Generate text (greedy or sampling)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=eos_token_id,
        )
    output_text = tokenizer.decode(output_ids[0].tolist())

    print(f"Prompt: {args.prompt}")
    print(f"Generated: {output_text}")

if __name__ == '__main__':
    main()
