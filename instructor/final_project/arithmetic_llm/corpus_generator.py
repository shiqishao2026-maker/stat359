"""Corpus generation module for arithmetic LLM training."""

import json
from typing import Tuple
from .generator import ExpressionGenerator
from .evaluator import eval_expression


class CorpusGenerator:
    """Generates training corpus of arithmetic expressions and evaluations."""
    
    def __init__(
        self,
        num_samples: int,
        max_depth: int = 5,
        num_range: Tuple[int, int] = (1, 20),
        invalid_rate: float = 0.1,
        output_path: str = "corpus.txt"
    ):
        """
        Initialize corpus generator.
        
        Args:
            num_samples: Number of expression-evaluation pairs to generate
            max_depth: Maximum depth of expression trees
            num_range: Range of numbers to use in expressions
            invalid_rate: Fraction of invalid expressions to include
            output_path: Path to save generated corpus
        """
        self.num_samples = num_samples
        self.max_depth = max_depth
        self.num_range = num_range
        self.invalid_rate = invalid_rate
        self.output_path = output_path
        self.generator = ExpressionGenerator(
            max_depth=max_depth,
            num_range=num_range,
            invalid_rate=invalid_rate
        )
    
    def generate_corpus(self) -> None:
        """Generate foundational training corpus and save to disk in JSONL format."""
        with open(self.output_path, 'w') as f:
            for _ in range(self.num_samples):
                expression = self.generator.generate()
                result = eval_expression(expression)
                
                # Create JSON object with structured data
                entry = {
                    'expression': result['expression'],
                    'problem': result['problem'],
                    'solution': result['solution'],
                    'answer': result['answer']
                }
                
                # Write as single line JSON
                f.write(json.dumps(entry) + '\n')
    
    def generate_instruction_corpus(self, output_path: str) -> None:
        """Generate instruction-formatted corpus for fine-tuning in JSONL format."""
        with open(output_path, 'w') as f:
            for _ in range(self.num_samples):
                expression = self.generator.generate()
                result = eval_expression(expression)
                
                # Create JSON object with structured data
                entry = {
                    'expression': result['expression'],
                    'problem': result['problem'],
                    'solution': result['solution'],
                    'answer': result['answer']
                }
                
                # Write as single line JSON
                f.write(json.dumps(entry) + '\n')


