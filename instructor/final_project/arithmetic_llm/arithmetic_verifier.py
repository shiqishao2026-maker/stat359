"""Verifier for arithmetic solutions used in GRPO training."""

from typing import Optional
import re


class ArithmeticVerifier:
    """Verifier for arithmetic solutions.

    Extracts final results from generated text and compares
    with ground truth to compute rewards.
    """

    def extract_final_result(self, generated_text: str) -> Optional[int]:
        """Extract final numeric result from generated text.

        Args:
            generated_text: Generated solution text

        Returns:
            Extracted integer result, or None if not found/parseable
        """
        error_match = re.search(
            r"Final Result\s*:\s*ERROR\b",
            generated_text,
            flags=re.IGNORECASE
        )
        if error_match:
            return None

        match = re.search(
            r"Final Result\s*:\s*([+-]?\s*\d+)",
            generated_text,
            flags=re.IGNORECASE
        )
        if match:
            raw_value = match.group(1).replace(" ", "")
            try:
                return int(raw_value)
            except ValueError:
                return None

        return None

    def compute_reward(self, generated_text: str, ground_truth: int) -> float:
        """Compute reward for generated response.

        Args:
            generated_text: Generated solution text
            ground_truth: Correct answer

        Returns:
            Reward value (1.0 if correct, 0.0 otherwise)
        """
        result = self.extract_final_result(generated_text)
        if result is None:
            return 0.0
        return 1.0 if result == ground_truth else 0.0
