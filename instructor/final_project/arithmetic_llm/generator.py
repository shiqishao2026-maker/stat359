import random


class ExpressionGenerator:
    def __init__(self, max_depth=2, num_range=(1, 20), invalid_rate=0.1):
        self.max_depth = max_depth
        self.num_range = num_range
        self.invalid_rate = invalid_rate

    

    def generate(self, current_depth=0):
        # If we reached max depth or a random chance, return a number
        if current_depth >= self.max_depth or random.random() < 0.3:
            return str(random.randint(*self.num_range))
        
        # Otherwise, expand into an operation
        op = random.choice(['+', '-'])
        left = self.generate(current_depth + 1)
        right = self.generate(current_depth + 1)

        if random.random() < self.invalid_rate:    
            error_type = random.choice(['missing_operand_right', 
                                        'missing_operand_left',
                                        'extra_operator++', 
                                        'extra_operator--',
                                        'unbalanced_paren_right',
                                        'unbalanced_paren_left',
                                        'arbitrary'
                                        ])
            if error_type == 'missing_operand_right':
                return f"{left} +"
            elif error_type == 'missing_operand_left':
                return f"+ {right}"
            elif error_type == 'extra_operator++':
                return f"{left} ++ {right}"
            elif error_type == 'extra_operator--':
                return f"{left} -- {right}"
            elif error_type == 'unbalanced_paren_right':
                return f"({left} {op} {right}"
            elif error_type == 'unbalanced_paren_left':
                return f"{left} {op} {right})"
            else:  # arbitrary error
                return self._generate_invalid()
        else:
            # Randomly decide to wrap in parentheses for visual structure
            if current_depth > 0:
                return f"({left} {op} {right})"
            return f"{left} {op} {right}"
    
    def _generate_invalid(self):
        # Keep numeric tokens within num_range, even for invalid expressions.
        tokens = ['+', '-', '(', ')']
        length = random.randint(2, 20)
        parts = []
        for _ in range(length):
            if random.random() < 0.4:
                parts.append(str(random.randint(*self.num_range)))
            else:
                parts.append(random.choice(tokens))
        # Avoid accidental digit concatenation across numeric tokens.
        out = []
        for i, tok in enumerate(parts):
            out.append(tok)
            if i < len(parts) - 1:
                next_tok = parts[i + 1]
                if tok.isdigit() and next_tok.isdigit():
                    out.append(' ')
                elif random.random() < 0.3:
                    out.append(' ')
        return ''.join(out)

if __name__ == "__main__":
    # Usage
    for _ in range(5):
        generator = ExpressionGenerator(max_depth=5, invalid_rate=0.1)
        new_expr = generator.generate()

        print(f"Generated Expression: {new_expr}")

    for _ in range(5):
        generator = ExpressionGenerator(max_depth=5, invalid_rate=-1.0)
        new_expr = generator.generate()

        print(f"Generated Expression: {new_expr}")
