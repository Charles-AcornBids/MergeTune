#!/usr/bin/env python3
"""
Example usage of optimization from optimize.py
"""
from optimize import run_regex_optimization, optimize

# Example 1: Use the full optimization workflow
# This will generate test data, benchmark both patterns, and show performance improvements
def example_full_workflow():
    print("\n" + "="*80)
    print("EXAMPLE 1: Full Regex Optimization Workflow")
    print("="*80)

    # Define a slow/inefficient regex pattern
    slow_pattern = r"(\d+\.)+\d+"

    # Run the complete optimization workflow
    run_regex_optimization(slow_pattern)


# Example 2: Just get the optimized pattern without benchmarking
def example_quick_optimize():
    print("\n" + "="*80)
    print("EXAMPLE 2: Quick Pattern Optimization (no benchmarking)")
    print("="*80)

    slow_pattern = r"(http|https)://.*"

    print(f"\nOriginal pattern: {slow_pattern}")

    # Get optimized pattern
    optimized_pattern = optimize(slow_pattern)

    print(f"\nOptimized pattern: {optimized_pattern}")


# Example 3: Optimize a complex email regex
def example_email_regex():
    print("\n" + "="*80)
    print("EXAMPLE 3: Email Regex Optimization")
    print("="*80)

    slow_pattern = r"(\w+)+@(\w+)+\.(\w+)+"

    run_regex_optimization(slow_pattern)


# Example 4: Optimize a phone number regex
def example_phone_regex():
    print("\n" + "="*80)
    print("EXAMPLE 4: Phone Number Regex Optimization")
    print("="*80)

    slow_pattern = r"(\d{1}|\d{2}|\d{3})-(\d{1}|\d{2}|\d{3})-(\d{1}|\d{2}|\d{3}|\d{4})"

    run_regex_optimization(slow_pattern)


# Example 5: Optimize Python code
def example_code_optimization():
    print("\n" + "="*80)
    print("EXAMPLE 5: Python Code Optimization")
    print("="*80)

    slow_code = """
def slow_function(n):
    result = []
    for i in range(n):
        temp = i * i
        result.append(temp)
    return result
"""

    print(f"\nOriginal code: {slow_code}")

    # Get optimized code
    optimized_code = optimize(slow_code)

    print(f"\nOptimized code:\n{optimized_code}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("OPTIMIZATION EXAMPLES")
    print("="*80)

    # Run all examples
    # Uncomment the ones you want to run:

    example_full_workflow()
    # example_quick_optimize()
    # example_email_regex()
    # example_phone_regex()
    # example_code_optimization()

    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETE")
    print("="*80 + "\n")
