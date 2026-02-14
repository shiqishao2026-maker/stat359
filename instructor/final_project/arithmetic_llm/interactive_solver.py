"""Interactive interface for arithmetic problem solving.

This module provides an interactive REPL interface for solving arithmetic
problems using the trained instruction-tuned model.
"""

import torch


class InteractiveArithmeticSolver:
    """Interactive solver for arithmetic problems using trained model.
    
    This class provides a REPL interface where users can enter arithmetic
    expressions and receive step-by-step solutions from the trained model.
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str = (
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )
    ):
        """Initialize interactive solver.
        
        Args:
            model_path: Path to instruction-tuned model checkpoint
            tokenizer_path: Path to tokenizer directory
            device: Device for inference ('cuda', 'mps', or 'cpu')
        """
        self.device = device
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        
        print("Loading model and tokenizer...")
        
        # Load tokenizer
        from .arithmetic_tokenizer import (
            ArithmeticBPETokenizer
        )
        self.tokenizer = ArithmeticBPETokenizer()
        self.tokenizer.load(tokenizer_path)
        
        # Load model
        from .transformer_model import (
            ArithmeticTransformer
        )
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model configuration from checkpoint
        if 'config' in checkpoint:
            config = checkpoint['config']
            self.model = ArithmeticTransformer(
                vocab_size=len(self.tokenizer.token2id),
                d_model=config.get('d_model', 256),
                nhead=config.get('nhead', 8),
                num_layers=config.get('num_layers', 6),
                dim_feedforward=config.get('dim_feedforward', 1024),
                dropout=config.get('dropout', 0.1),
                max_seq_length=config.get('max_seq_length', 512)
            )
        else:
            # Fallback to default configuration
            self.model = ArithmeticTransformer(
                vocab_size=len(self.tokenizer.token2id)
            )
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(device)
        self.model.eval()
        
        print(f"Model loaded successfully on {device}")
        print(f"Vocabulary size: {len(self.tokenizer.token2id)}")
    
    def run(self) -> None:
        """Start interactive REPL loop.
        
        Displays welcome message and continuously prompts user for arithmetic
        expressions until they choose to exit.
        """
        print("\n" + "=" * 60)
        print("ARITHMETIC LLM - INTERACTIVE SOLVER")
        print("=" * 60)
        print("\nWelcome! Enter arithmetic expressions to solve.")
        print("The model will show step-by-step reasoning.")
        print("\nSupported operations: + (addition), - (subtraction)")
        print("Example: 5 + (10 - 3)")
        print("\nCommands:")
        print("  - Type 'exit' or 'quit' to exit")
        print("  - Press Ctrl+C to exit")
        print("=" * 60 + "\n")
        
        while True:
            try:
                # Get user input
                expression = input("Enter expression: ").strip()
                
                # Check for exit commands
                if expression.lower() in ['exit', 'quit', 'q']:
                    print("\nThank you for using Arithmetic LLM!")
                    break
                
                # Skip empty input
                if not expression:
                    continue
                
                # Solve the expression
                try:
                    solution = self.solve(expression)
                    formatted_output = self.format_output(solution)
                    print(formatted_output)
                except Exception as e:
                    print("\nError: Failed to generate solution.")
                    print(f"Details: {str(e)}")
                    print("Please try again with a different expression.\n")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                print("Thank you for using Arithmetic LLM!")
                break
            except EOFError:
                print("\n\nEnd of input.")
                print("Thank you for using Arithmetic LLM!")
                break
    
    def solve(self, expression: str) -> str:
        """Solve arithmetic expression.
        
        Args:
            expression: Arithmetic expression to evaluate
            
        Returns:
            Generated solution with reasoning steps
        """
        # Format expression as instruction prompt
        prompt = f"Evaluate: {expression} <think>"
        
        # Encode prompt (with BOS, without EOS since we're generating)
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        # Remove EOS token if present (we want to continue generating)
        eos_token_id = self.tokenizer.token2id.get('<eos>', None)
        if eos_token_id is not None and input_ids and input_ids[-1] == eos_token_id:
            input_ids = input_ids[:-1]
        
        input_tensor = torch.tensor(
            [input_ids],
            dtype=torch.long
        ).to(self.device)
        
        # Generate solution
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_tensor,
                max_length=256,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                eos_token_id=eos_token_id
            )
        
        # Decode generated text (skip special tokens for cleaner output)
        generated_text = self.tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
        
        return generated_text
    
    def format_output(self, generated_text: str) -> str:
        """Format model output for display.
        
        Args:
            generated_text: Generated text from model
            
        Returns:
            Formatted output string for display
        """
        output = "\n" + "-" * 60 + "\n"
        output += "SOLUTION:\n"
        output += "-" * 60 + "\n"
        
        # Split text into lines
        lines = generated_text.split('\n')
        
        # Track if we found reasoning and result
        found_reasoning = False
        found_result = False
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Handle <think> tag
            if line == '<think>':
                output += "\nReasoning Steps:\n"
                found_reasoning = True
                continue
            
            # Handle </think> tag
            if line == '</think>':
                output += "\n"
                continue
            
            # Handle step lines
            if line.startswith('Step '):
                output += f"  {line}\n"
                found_reasoning = True
                continue
            
            # Handle expression now lines
            if line.startswith('Expression now:'):
                output += f"  {line}\n"
                found_reasoning = True
                continue
            
            # Handle final result
            if line.startswith('Final Result:'):
                output += f"\n{line}\n"
                found_result = True
                continue
            
            # Handle other lines (like the original prompt)
            if line.startswith('Evaluate:'):
                continue
            
            # Add any other content
            output += f"{line}\n"
        
        # Check if output is complete
        if not found_reasoning and not found_result:
            output += "\nWarning: Model output may be incomplete or malformed.\n"
            output += "Raw output:\n"
            output += generated_text + "\n"
        
        output += "-" * 60 + "\n"
        
        return output


def main():
    """Main entry point for interactive solver."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Interactive arithmetic problem solver'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to instruction-tuned model checkpoint'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        required=True,
        help='Path to tokenizer directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for inference (cuda or cpu)'
    )
    
    args = parser.parse_args()
    
    # Create and run interactive solver
    solver = InteractiveArithmeticSolver(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        device=args.device
    )
    solver.run()


if __name__ == '__main__':
    main()


