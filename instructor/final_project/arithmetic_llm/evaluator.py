"""Model evaluation module for arithmetic LLM.

This module provides evaluation capabilities for trained arithmetic models,
including accuracy computation, result extraction, and reasoning verification.
"""

import re
import os
import json
import torch
from typing import Dict, Optional, List, Tuple, Union
from datetime import datetime


class Node:
    def __init__(self, kind, value=None, left=None, right=None, op=None):
        self.kind = kind
        self.value = value
        self.left = left
        self.right = right
        self.op = op
        self.evaluated = kind == 'num'

class ArithmeticEvaluator:
    def __init__(self, expression):
        # Check for invalid consecutive numbers before removing spaces
        # Pattern: digit(s), whitespace(s), digit(s) without an operator between
        if re.search(r'\d\s+\d', expression):
            raise ValueError("Invalid syntax: consecutive numbers separated by whitespace")
        
        self.expression = expression.replace(" ", "")
        self.tokens = re.findall(r'\d+|[+\-()=]', self.expression)
        self.pos = 0
        self.steps = []
        self.root = None

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self):
        token = self.peek()
        self.pos += 1
        return token

    def expect(self, expected):
        token = self.consume()
        if token != expected:
            raise ValueError(f"Expected '{expected}', got '{token}'")
        return token

    def parse_expression(self):
        # Handles + and -
        node = self.parse_term()
        while self.peek() in ('+', '-'):
            op = self.consume()
            right = self.parse_term()
            node = Node('bin', left=node, right=right, op=op)
        return node

    def parse_term(self):
        # Handles parentheses and numbers
        token = self.consume()
        if token is None:
            raise ValueError("Unexpected end of expression")
        if token == '(':
            result = self.parse_expression()
            self.expect(')')
            return result
        if token in ('+', '-', ')'):
            raise ValueError(f"Unexpected token '{token}'")
        if not token.isdigit():
            raise ValueError(f"Invalid token '{token}'")
        return Node('num', value=int(token))

    def render_expression(self, node, is_root=False):
        if node.evaluated:
            return str(node.value)
        expr = f"{self.render_expression(node.left)} {node.op} {self.render_expression(node.right)}"
        return expr if is_root else f"({expr})"

    def evaluate_node(self, node):
        if node.kind == 'num':
            return node.value
        left_val = self.evaluate_node(node.left)
        right_val = self.evaluate_node(node.right)
        node.value = (left_val + right_val) if node.op == '+' else (left_val - right_val)
        node.evaluated = True
        expr_now = self.render_expression(self.root, is_root=True)
        self.steps.append((f"{left_val} {node.op} {right_val} = {node.value}", expr_now))
        return node.value

    def evaluate(self):
        self.steps = []
        self.root = self.parse_expression()
        if self.peek() is not None:
            raise ValueError(f"Unexpected token '{self.peek()}' at position {self.pos}")
        final_result = self.evaluate_node(self.root)
        return final_result, self.steps


def eval_expression(expression):
    try:
        evaluator = ArithmeticEvaluator(expression)
        result, steps = evaluator.evaluate()
        problem_statement = f"Evaluate: {expression}"
        answers = ['<think>']
        for i, (step, expr_now) in enumerate(steps, 1):
            answers.append(f"Step {i}: {step}")
            answers.append(f"Expression now: {expr_now}")
        answers.append('</think>')
        answers.append(f"Final Result: {result}")
        return {
            'expression': expression,
            'problem': problem_statement,
            'solution': "\n".join(answers),
            'answer': result

        }
    except Exception:
        return {
            'expression': expression,
            'problem': f"Evaluate: {expression}",
            'solution': "<think>Evaluation ERROR</think>\nFinal Result: ERROR",
            'answer': "ERROR"
        }



if __name__ == "__main__":
    print(eval_expression("5 + (10 - 3)"))
    print(eval_expression("((((15 - 5) + 8) - (14 - (16 + 10))) + 1) - ((((4 - 11) - (15 - 13)) + (6 - (5 - 4))) - 11)"))

    from .generator import ExpressionGenerator
    for _ in range(5):
        generator = ExpressionGenerator(max_depth=5, invalid_rate=0.1)
        new_expr = generator.generate()

        print(f"\n\nGenerated Expression: {new_expr}\n")
        print(eval_expression(new_expr))



class ModelEvaluator:
    """Evaluator for trained arithmetic models.
    
    This class loads a trained model and tokenizer, generates test expressions,
    and computes accuracy metrics by comparing model outputs to ground truth.
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        base_checkpoint_path: Optional[str] = None,
        device: str = (
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )
    ):
        """Initialize evaluator with trained model.
        
        Args:
            model_path: Path to model checkpoint
            tokenizer_path: Path to tokenizer directory
            device: Device for inference ('cuda', 'mps', or 'cpu')
        """
        self.device = device
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.base_checkpoint_path = base_checkpoint_path
        
        # Load tokenizer
        from .arithmetic_tokenizer import ArithmeticBPETokenizer
        self.tokenizer = ArithmeticBPETokenizer()
        self.tokenizer.load(tokenizer_path)
        
        # Load model
        from .transformer_model import ArithmeticTransformer
        checkpoint = torch.load(model_path, map_location="cpu")
        is_adapter = (
            isinstance(checkpoint, dict)
            and "lora_state" in checkpoint
            and "metadata" in checkpoint
        )

        if is_adapter:
            metadata = checkpoint["metadata"]
            base_path = self.base_checkpoint_path or metadata.get("base_model_path")
            if not base_path:
                raise ValueError(
                    "Adapter checkpoint provided without base model. "
                    "Pass --base-checkpoint or use a merged model."
                )
            base_checkpoint = torch.load(base_path, map_location="cpu")
            model_config = base_checkpoint.get("model_config")
            if model_config is None:
                base_config = base_checkpoint.get("config", {})
                model_config = {
                    "vocab_size": len(self.tokenizer.token2id),
                    "d_model": base_config.get("d_model", 256),
                    "nhead": base_config.get("nhead", 8),
                    "num_layers": base_config.get("num_layers", 6),
                    "dim_feedforward": base_config.get("dim_feedforward", 1024),
                    "dropout": base_config.get("dropout", 0.1),
                    "max_seq_length": base_config.get("max_seq_length", 512),
                }
            else:
                model_config = dict(model_config)
                model_config["vocab_size"] = len(self.tokenizer.token2id)

            self.model = ArithmeticTransformer(**model_config)
            if "model_state_dict" in base_checkpoint:
                self.model.load_state_dict(base_checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(base_checkpoint)

            self.model.load_lora_adapters(model_path)
        else:
            if 'model_config' in checkpoint:
                model_config = dict(checkpoint['model_config'])
                model_config["vocab_size"] = len(self.tokenizer.token2id)
                self.model = ArithmeticTransformer(**model_config)
            elif 'config' in checkpoint:
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
    
    def evaluate(
        self,
        num_samples: int = 1000,
        max_depth: int = 5,
        num_range: Tuple[int, int] = (1, 20),
        output_dir: Optional[str] = None,
        batch_size: int = 32,
        max_gen_length: int = 256
    ) -> Dict[str, float]:
        """Evaluate model on test set.
        
        Args:
            num_samples: Number of test expressions to generate
            max_depth: Maximum depth of test expressions
            num_range: Range of numbers to use in expressions
            output_dir: Optional directory to save evaluation results
            batch_size: Batch size for inference (default: 32)
            max_gen_length: Maximum generation length in tokens (default: 256)
            
        Returns:
            Dictionary with accuracy metrics:
                - exact_match_accuracy: Percentage of correct final results
                - parse_success_rate: Percentage of parseable outputs
                - avg_generation_length: Average number of tokens generated
                - total_samples: Total number of test samples
        """
        from .generator import ExpressionGenerator
        
        # Generate test set
        generator = ExpressionGenerator(
            max_depth=max_depth,
            num_range=num_range,
            invalid_rate=0.0  # Only valid expressions for evaluation
        )
        
        test_expressions = []
        test_answers = []
        
        print(f"Generating {num_samples} test expressions...")
        for _ in range(num_samples):
            expression = generator.generate()
            
            # Get ground truth answer
            result = eval_expression(expression)
            if result['answer'] != 'ERROR':
                test_expressions.append(expression)
                test_answers.append(result['answer'])
        
        print(f"Generated {len(test_expressions)} valid test expressions")
        
        # Evaluate model on test set using batches
        correct = 0
        parseable = 0
        total_length = 0
        sample_outputs = []
        
        print(f"Evaluating model with batch size {batch_size}...")
        
        # Process in batches
        for batch_start in range(0, len(test_expressions), batch_size):
            batch_end = min(batch_start + batch_size, len(test_expressions))
            batch_expressions = test_expressions[batch_start:batch_end]
            batch_answers = test_answers[batch_start:batch_end]
            
            # Format prompts for batch
            batch_prompts = [f"Evaluate: {expr} <think>" for expr in batch_expressions]
            
            # Generate solutions for batch
            batch_generated_texts = self._generate_batch(batch_prompts, max_length=max_gen_length)
            
            # Process batch results
            for i, (expression, ground_truth, generated_text) in enumerate(
                zip(batch_expressions, batch_answers, batch_generated_texts)
            ):
                total_length += len(generated_text.split())
                
                # Extract final result
                predicted_result = self.extract_final_result(generated_text)
                
                # Check if output is parseable
                if predicted_result is not None:
                    parseable += 1
                    
                    # Check if result is correct
                    if predicted_result == ground_truth:
                        correct += 1
                
                # Save sample outputs (first 10)
                global_idx = batch_start + i
                if global_idx < 10:
                    sample_outputs.append({
                        'expression': expression,
                        'ground_truth': ground_truth,
                        'predicted': predicted_result,
                        'generated_text': generated_text,
                        'correct': predicted_result == ground_truth if predicted_result is not None else False
                    })
            
            # Progress update
            print(f"Evaluated {batch_end}/{len(test_expressions)} samples")
        
        # Calculate metrics
        total = len(test_expressions)
        metrics = {
            'exact_match_accuracy': (correct / total * 100) if total > 0 else 0.0,
            'parse_success_rate': (parseable / total * 100) if total > 0 else 0.0,
            'avg_generation_length': (total_length / total) if total > 0 else 0.0,
            'total_samples': total,
            'correct_samples': correct,
            'parseable_samples': parseable
        }
        
        # Save results if output directory is provided
        if output_dir:
            self._save_results(metrics, sample_outputs, output_dir)
        
        return metrics
    
    def _generate_solution(self, prompt: str, max_length: int = 256) -> str:
        """Generate solution for a given prompt.
        
        Args:
            prompt: Input prompt text
            max_length: Maximum generation length
            
        Returns:
            Generated text
        """
        # Encode prompt (with BOS, without EOS since we're generating)
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        # Remove EOS token if present (we want to continue generating)
        eos_token_id = self.tokenizer.token2id.get('<eos>', None)
        if eos_token_id is not None and input_ids and input_ids[-1] == eos_token_id:
            input_ids = input_ids[:-1]
        
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_tensor,
                max_length=max_length,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                eos_token_id=eos_token_id
            )
        
        # Decode (skip special tokens for cleaner output)
        generated_text = self.tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
        
        return generated_text
    
    def _generate_batch(self, prompts: List[str], max_length: int = 256) -> List[str]:
        """Generate solutions for a batch of prompts.
        
        Args:
            prompts: List of input prompt texts
            max_length: Maximum generation length
            
        Returns:
            List of generated texts
        """
        # Encode all prompts (with BOS, without EOS since we're generating)
        batch_input_ids = []
        eos_token_id = self.tokenizer.token2id.get('<eos>', None)
        
        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            # Remove EOS token if present (we want to continue generating)
            if eos_token_id is not None and input_ids and input_ids[-1] == eos_token_id:
                input_ids = input_ids[:-1]
            batch_input_ids.append(input_ids)
        
        # Left-pad sequences so last token aligns across batch
        max_input_len = max(len(ids) for ids in batch_input_ids)
        pad_token_id = self.tokenizer.token2id.get('<pad>', 0)
        
        padded_input_ids = []
        attention_masks = []
        for ids in batch_input_ids:
            pad_len = max_input_len - len(ids)
            padded = [pad_token_id] * pad_len + ids
            mask = [0] * pad_len + [1] * len(ids)
            padded_input_ids.append(padded)
            attention_masks.append(mask)
        
        # Create batch tensor
        input_tensor = torch.tensor(padded_input_ids, dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.float).to(self.device)
        
        # Generate for batch
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_tensor,
                max_length=max_length,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                eos_token_id=eos_token_id,
                attention_mask=attention_mask
            )
        
        # Decode all outputs (skip special tokens for cleaner output)
        generated_texts = []
        for ids in generated_ids:
            text = self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def extract_final_result(self, generated_text: str) -> Optional[Union[int, str]]:
        """Extract final answer from model output.
        
        Args:
            generated_text: Generated text from model
            
        Returns:
            Extracted integer result, "ERROR" if explicitly present, or None if not found/parseable
        """
        # Allow explicit ERROR results
        error_match = re.search(r'Final Result\s*:\s*ERROR\b', generated_text, flags=re.IGNORECASE)
        if error_match:
            return "ERROR"

        # Look for "Final Result: N" pattern with flexible spacing and sign
        match = re.search(r'Final Result\s*:\s*([+-]?\s*\d+)', generated_text, flags=re.IGNORECASE)
        if match:
            raw_value = match.group(1).replace(" ", "")
            try:
                return int(raw_value)
            except ValueError:
                return None
        
        return None
    
    def verify_reasoning_steps(
        self,
        expression: str,
        generated_text: str
    ) -> bool:
        """Verify that reasoning steps are mathematically correct.
        
        Args:
            expression: Original arithmetic expression
            generated_text: Generated solution text
            
        Returns:
            True if all reasoning steps are correct, False otherwise
        """
        # Extract all step patterns: "Step N: A op B = C"
        step_pattern = r'Step \d+:\s*(-?\d+)\s*([+\-])\s*(-?\d+)\s*=\s*(-?\d+)'
        steps = re.findall(step_pattern, generated_text)
        
        if not steps:
            # No steps found
            return False
        
        # Verify each step
        for step in steps:
            left, op, right, result = step
            try:
                left_val = int(left)
                right_val = int(right)
                result_val = int(result)
                
                # Compute expected result
                if op == '+':
                    expected = left_val + right_val
                elif op == '-':
                    expected = left_val - right_val
                else:
                    return False
                
                # Check if step is correct
                if expected != result_val:
                    return False
            except ValueError:
                return False
        
        return True
    
    def _save_results(
        self,
        metrics: Dict[str, float],
        sample_outputs: List[Dict],
        output_dir: str
    ) -> None:
        """Save evaluation results to disk.
        
        Args:
            metrics: Dictionary of evaluation metrics
            sample_outputs: List of sample output dictionaries
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_path = os.path.join(output_dir, f'evaluation_metrics_{timestamp}.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save sample outputs
        samples_path = os.path.join(output_dir, f'sample_outputs_{timestamp}.json')
        with open(samples_path, 'w') as f:
            json.dump(sample_outputs, f, indent=2)
        
        # Save summary text file
        summary_path = os.path.join(output_dir, f'evaluation_summary_{timestamp}.txt')
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("ARITHMETIC LLM EVALUATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Tokenizer: {self.tokenizer_path}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write("METRICS:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total Samples: {metrics['total_samples']}\n")
            f.write(f"Correct Samples: {metrics['correct_samples']}\n")
            f.write(f"Parseable Samples: {metrics['parseable_samples']}\n")
            f.write(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.2f}%\n")
            f.write(f"Parse Success Rate: {metrics['parse_success_rate']:.2f}%\n")
            f.write(f"Avg Generation Length: {metrics['avg_generation_length']:.2f} tokens\n\n")
            f.write("SAMPLE OUTPUTS:\n")
            f.write("-" * 60 + "\n")
            for i, sample in enumerate(sample_outputs, 1):
                f.write(f"\nSample {i}:\n")
                f.write(f"  Expression: {sample['expression']}\n")
                f.write(f"  Ground Truth: {sample['ground_truth']}\n")
                f.write(f"  Predicted: {sample['predicted']}\n")
                f.write(f"  Correct: {sample['correct']}\n")
                f.write("  Generated Text:\n")
                for line in sample['generated_text'].split('\n'):
                    f.write(f"    {line}\n")
        
        print(f"\nEvaluation results saved to {output_dir}")
        print(f"  - Metrics: {metrics_path}")
        print(f"  - Samples: {samples_path}")
        print(f"  - Summary: {summary_path}")


