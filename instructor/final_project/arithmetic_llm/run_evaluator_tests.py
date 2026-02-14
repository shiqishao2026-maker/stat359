#!/usr/bin/env python3
"""Convenience script to run all evaluator tests."""

import sys
import subprocess


def run_tests():
    """Run all evaluator tests and display results."""
    print("=" * 70)
    print("Running Arithmetic Evaluator Tests")
    print("=" * 70)
    print()
    
    # Run pytest with both test files
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_evaluator_unit.py",
        "tests/test_evaluator_properties.py",
        "-v",
        "--tb=short",
        "--color=yes"
    ]
    
    result = subprocess.run(cmd)
    
    print()
    print("=" * 70)
    if result.returncode == 0:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed. See output above for details.")
    print("=" * 70)
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(run_tests())
