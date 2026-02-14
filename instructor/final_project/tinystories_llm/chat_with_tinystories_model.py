import os
import torch
import argparse
from bpe_tokenizer import BPETokenizer
from transformer_model import TinyStoriesConfig, TinyStoriesForCausalLM
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Chat with a TinyStories chat model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint (e.g., final_model.pth)")
    parser.add_argument("--tokenizer_path", type=str, default="bpe_tokenizer_tinystories.pkl", help="Path to BPE tokenizer")
    parser.add_argument("--max_seq_len", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size of the model")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--intermediate_size", type=int, default=1024, help="Size of the intermediate (feed-forward) layer")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--window_size", type=int, default=256, help="Attention window size for local attention")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device to use: auto, cpu, cuda, or mps")
    parser.add_argument("--user_token", type=str, default="<user>", help="Token to represent user messages")
    parser.add_argument("--assistant_token", type=str, default="<assistant>", help="Token to represent assistant messages")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling top-p")
    parser.add_argument("--max_gen_len", type=int, default=100, help="Max tokens to generate in response")
    return parser.parse_args()

def get_device(device_preference):
    if device_preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif device_preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            raise ValueError("CUDA requested but not available.")
    elif device_preference == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        else:
            raise ValueError("MPS requested but not available.")
    else:
        return torch.device("cpu")

def load_tokenizer(tokenizer_path):
    return BPETokenizer.load(tokenizer_path)

def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer_path)
    user_token_id = tokenizer.token2id.get(args.user_token, None)
    assistant_token_id = tokenizer.token2id.get(args.assistant_token, None)

    # Model config (should match training)
    config = TinyStoriesConfig(
        vocab_size=len(tokenizer.token2id),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        max_position_embeddings=args.max_seq_len,
        window_size=args.window_size,
    )
    model = TinyStoriesForCausalLM(config).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    print("\nWelcome to TinyStories Chat! Type 'exit' to quit.\n")
    conversation_history = []
    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            print("\n[Session ended: EOF received]")
            break
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        # Append user turn
        conversation_history.append((args.user_token, user_input))
        # Build prompt
        prompt = ""
        for role, text in conversation_history:
            prompt += f"{role} {text} "
        prompt += f"{args.assistant_token}"
        input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long).to(device)
        
        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + args.max_gen_len,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        output_ids = output_ids[:, input_ids.shape[1]:][0]
        stop_idx = -1
        for i, output_id in enumerate(output_ids):
            if output_id == user_token_id or output_id == assistant_token_id:
                stop_idx = i
                break
        if stop_idx > 0:
            output_ids = output_ids[:stop_idx]
        output_text = tokenizer.decode(output_ids.tolist(),remove_special_tokens=False)
        # Extract assistant's response (stop at next <user> or <assistant> token)
        response = ""
        if args.assistant_token in output_text:
            after_assistant = output_text.split(args.assistant_token, 1)[1]
            # Stop at next special token
            for stop_token in [args.user_token, args.assistant_token]:
                idx = after_assistant.find(stop_token)
                if idx != -1:
                    after_assistant = after_assistant[:idx]
            response = after_assistant.strip()
        else:
            response = output_text.strip()
        print(f"Assistant: {response}\n")
        # Append assistant turn
        conversation_history.append((args.assistant_token, response))

if __name__ == "__main__":
    main()
