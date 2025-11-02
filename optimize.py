import timeit
import statistics
import inspect
import os
import re
from typing import Callable, Any, Dict, List, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from inquirer import List as InquirerList, prompt
from inquirer.themes import GreenPassion

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_default_values(code_func: Callable, **existing_kwargs) -> Dict[str, Any]:
    """
    Generate default values for function parameters that don't have them using GPT-4o-mini.

    Args:
        code_func: The function to analyze
        **existing_kwargs: Any already provided keyword arguments

    Returns:
        Dictionary of parameter names to generated default values
    """
    sig = inspect.signature(code_func)
    missing_params = []

    # Find parameters without defaults that aren't in existing_kwargs
    for param_name, param in sig.parameters.items():
        if param.default == inspect.Parameter.empty and param_name not in existing_kwargs:
            # Get type hint if available
            type_hint = param.annotation if param.annotation != inspect.Parameter.empty else "Any"
            missing_params.append({
                "name": param_name,
                "type": str(type_hint)
            })

    if not missing_params:
        return existing_kwargs

    # Get function source code for context
    try:
        func_source = inspect.getsource(code_func)
    except (OSError, TypeError):
        func_source = f"Function name: {code_func.__name__}"

    # Create prompt for GPT-4o-mini
    prompt = f"""Given this Python function:

{func_source}

Generate appropriate default values for these parameters that have no defaults:
{missing_params}

Return ONLY a Python dictionary in this exact format (no markdown, no explanation):
{{"param_name": value, "param_name2": value2}}

Use realistic, reasonable values that would make sense for benchmarking this function.
For numeric parameters, use small to medium-sized values (e.g., 10-1000).
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates Python default values. Return only valid Python dictionary syntax, no markdown formatting."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        generated_values_str = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if generated_values_str.startswith("```"):
            lines = generated_values_str.split("\n")
            generated_values_str = "\n".join([line for line in lines if not line.startswith("```")])
            generated_values_str = generated_values_str.strip()

        # Safely evaluate the dictionary
        generated_values = eval(generated_values_str)

        # Merge with existing kwargs
        result = {**generated_values, **existing_kwargs}

        print(f"\n[AI Generated Default Values for {code_func.__name__}]")
        for param in missing_params:
            if param["name"] in generated_values:
                print(f"  {param['name']}: {generated_values[param['name']]}")
        print()

        return result

    except Exception as e:
        print(f"Warning: Could not generate default values: {e}")
        print("Proceeding with existing kwargs only...")
        return existing_kwargs


def benchmark_code(
    code_func: Callable,
    number: int = 1000,
    repeat: int = 5,
    setup: str = "",
    auto_generate_defaults: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Benchmark a code block and return timing statistics.

    Args:
        code_func: A callable function to benchmark
        number: Number of times to execute the code in each repeat (default: 1000)
        repeat: Number of times to repeat the benchmark (default: 5)
        setup: Setup code to run before timing (default: "")
        auto_generate_defaults: If True, use AI to generate missing default values (default: True)
        **kwargs: Additional keyword arguments to pass to the function

    Returns:
        Dictionary containing timing statistics:
        - mean: Average execution time
        - median: Median execution time
        - min: Minimum execution time
        - max: Maximum execution time
        - std_dev: Standard deviation
        - total_runs: Total number of executions
        - timings: List of all timing results
    """
    # Auto-generate missing default values if enabled
    if auto_generate_defaults:
        kwargs = generate_default_values(code_func, **kwargs)

    # Create a wrapper that calls the function with kwargs
    def wrapped():
        return code_func(**kwargs) if kwargs else code_func()

    # Run the benchmark
    timings = timeit.repeat(wrapped, setup=setup, number=number, repeat=repeat)

    # Calculate statistics
    mean_time = statistics.mean(timings)
    median_time = statistics.median(timings)
    min_time = min(timings)
    max_time = max(timings)
    std_dev = statistics.stdev(timings) if len(timings) > 1 else 0

    return {
        "mean": mean_time,
        "median": median_time,
        "min": min_time,
        "max": max_time,
        "std_dev": std_dev,
        "total_runs": number * repeat,
        "timings": timings
    }


def benchmark_string_code(
    code: str,
    setup: str = "",
    number: int = 1000,
    repeat: int = 5
) -> Dict[str, Any]:
    """
    Benchmark code provided as a string.

    Args:
        code: Python code to benchmark as a string
        setup: Setup code to run before timing (default: "")
        number: Number of times to execute the code in each repeat (default: 1000)
        repeat: Number of times to repeat the benchmark (default: 5)

    Returns:
        Dictionary containing timing statistics
    """
    timings = timeit.repeat(code, setup=setup, number=number, repeat=repeat)

    mean_time = statistics.mean(timings)
    median_time = statistics.median(timings)
    min_time = min(timings)
    max_time = max(timings)
    std_dev = statistics.stdev(timings) if len(timings) > 1 else 0

    return {
        "mean": mean_time,
        "median": median_time,
        "min": min_time,
        "max": max_time,
        "std_dev": std_dev,
        "total_runs": number * repeat,
        "timings": timings
    }


def print_benchmark_results(name: str, results: Dict[str, Any]):
    """Pretty print benchmark results."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"{'='*60}")
    print(f"Total runs: {results['total_runs']:,}")
    print(f"Mean time:   {results['mean']:.6f} seconds")
    print(f"Median time: {results['median']:.6f} seconds")
    print(f"Min time:    {results['min']:.6f} seconds")
    print(f"Max time:    {results['max']:.6f} seconds")
    print(f"Std dev:     {results['std_dev']:.6f} seconds")
    print(f"{'='*60}")


def _is_regex_pattern(text: str) -> bool:
    """
    Detect if a string is likely a regex pattern.
    """
    # If it's short and contains regex metacharacters, likely a pattern
    if len(text) < 200 and any(char in text for char in ['+', '*', '?', '|', '[', ']', '{', '}', '^', '$']):
        # Try to compile it as regex
        try:
            re.compile(text)
            return True
        except Exception:
            pass

    return False


def optimize(code: str) -> str:
    """
    Optimize code or regex patterns.

    Args:
        code: The code or regex pattern to optimize

    Returns:
        Optimized version
    """
    # Detect if input is a regex pattern
    is_regex = _is_regex_pattern(code)

    if is_regex:
        prompt = f"""Optimize this:

{code}

Return only the optimized version, nothing else."""
    else:
        prompt = f"""Optimize this:

```python
{code}
```

Return only the optimized code, nothing else."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )

        optimized = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if optimized.startswith("```"):
            lines = optimized.split("\n")
            optimized = "\n".join([line for line in lines if not line.startswith("```")])
            optimized = optimized.strip()

        # Remove language identifier if present
        if optimized.startswith("python\n"):
            optimized = optimized[7:]

        # Remove quotes from regex patterns
        if is_regex:
            optimized = optimized.strip('"\'`')

        return optimized

    except Exception as e:
        print(f"Error: {e}")
        return code


def generate_regex_test_data(pattern: str, num_samples: int = 100) -> List[str]:
    """
    Generate test data for a regex pattern.

    Args:
        pattern: The regex pattern
        num_samples: Number of test strings to generate

    Returns:
        List of test strings (mix of matching and non-matching)
    """
    prompt = f"""Generate {num_samples} realistic test strings for this pattern:

{pattern}

Return only a Python list of strings."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=2000
        )

        test_data_str = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if test_data_str.startswith("```"):
            lines = test_data_str.split("\n")
            test_data_str = "\n".join([line for line in lines if not line.startswith("```")])
            test_data_str = test_data_str.strip()

        # Remove 'python' language identifier if present
        if test_data_str.startswith("python\n"):
            test_data_str = test_data_str[7:]

        test_data = eval(test_data_str)
        return test_data

    except Exception as e:
        print(f"Error generating test data: {e}")
        # Fallback: generate simple test data
        return [f"test{i}" for i in range(num_samples)]


def generate_regex_test_cases(pattern: str, num_positive: int = 25, num_negative: int = 25) -> Tuple[List[str], List[str]]:
    """
    Generate positive and negative test cases for a regex pattern.

    Args:
        pattern: The regex pattern
        num_positive: Number of strings that should match (default: 25)
        num_negative: Number of strings that should NOT match (default: 25)

    Returns:
        Tuple of (positive_cases, negative_cases)
    """
    prompt = f"""Generate test cases for this regex pattern: {pattern}

Create TWO separate lists:
1. {num_positive} strings that SHOULD match the pattern
2. {num_negative} strings that should NOT match the pattern

Return ONLY a Python tuple of two lists in this exact format:
(
    ["positive1", "positive2", ...],
    ["negative1", "negative2", ...]
)

Make the test cases realistic and diverse."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=2000
        )

        test_data_str = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if test_data_str.startswith("```"):
            lines = test_data_str.split("\n")
            test_data_str = "\n".join([line for line in lines if not line.startswith("```")])
            test_data_str = test_data_str.strip()

        # Remove 'python' language identifier if present
        if test_data_str.startswith("python\n"):
            test_data_str = test_data_str[7:]

        positive_cases, negative_cases = eval(test_data_str)
        return positive_cases, negative_cases

    except Exception as e:
        print(f"Error generating test cases: {e}")
        # Fallback: generate simple test data
        return (
            [f"match{i}" for i in range(num_positive)],
            [f"nomatch{i}" for i in range(num_negative)]
        )




def benchmark_regex(
    pattern: str,
    test_data: List[str],
    number: int = 100,
    repeat: int = 5
) -> Tuple[Dict[str, Any], List[bool]]:
    """
    Benchmark a regex pattern against test data.

    Args:
        pattern: The regex pattern to test
        test_data: List of test strings
        number: Number of times to execute in each repeat
        repeat: Number of times to repeat the benchmark

    Returns:
        Tuple of (timing statistics dict, list of match results)
    """
    compiled_pattern = re.compile(pattern)

    def run_matches():
        results = []
        for text in test_data:
            results.append(bool(compiled_pattern.search(text)))
        return results

    # Get match results for validation
    match_results = run_matches()

    # Benchmark
    timings = timeit.repeat(run_matches, number=number, repeat=repeat)

    mean_time = statistics.mean(timings)
    median_time = statistics.median(timings)
    min_time = min(timings)
    max_time = max(timings)
    std_dev = statistics.stdev(timings) if len(timings) > 1 else 0

    stats = {
        "mean": mean_time,
        "median": median_time,
        "min": min_time,
        "max": max_time,
        "std_dev": std_dev,
        "total_runs": number * repeat,
        "timings": timings
    }

    return stats, match_results


# Example functions to benchmark (INTENTIONALLY SLOW)
def slow_nested_loops(n: int = 100):
    """SLOW: Using nested loops instead of list comprehension."""
    result = []
    for i in range(n):
        temp = i * i
        result.append(temp)
    # Simulate more inefficiency
    final = []
    for item in result:
        final.append(item)
    return final


def slow_string_concat(n: int = 500):
    """SLOW: String concatenation with + operator."""
    result = ""
    for i in range(n):
        result = result + str(i) + ","
    return result


def slow_list_search(n: int = 1000):
    """SLOW: Searching in list repeatedly (O(n) each time)."""
    data = list(range(n))
    found = []
    search_terms = list(range(0, n, 10))
    for term in search_terms:
        if term in data:  # O(n) operation
            found.append(term)
    return found


def slow_dict_building(n: int = 1000):
    """SLOW: Building dict inefficiently."""
    result = {}
    for i in range(n):
        key = str(i)
        value = i * 2
        result[key] = value
    # Rebuild it for no reason
    final = {}
    for k in result.keys():
        final[k] = result[k]
    return final


def slow_filtering(n: int = 1000):
    """SLOW: Filtering with nested loops."""
    numbers = list(range(n))
    result = []
    for num in numbers:
        if num % 2 == 0:
            if num % 3 == 0:
                result.append(num)
    return result


def slow_sum_calculation(n: int = 1000):
    """SLOW: Summing inefficiently."""
    numbers = []
    for i in range(n):
        numbers.append(i)

    total = 0
    for num in numbers:
        total = total + num
    return total


def slow_unique_values(n: int = 500):
    """SLOW: Finding unique values using list."""
    data = [i % 50 for i in range(n)]  # Creates duplicates
    unique = []
    for item in data:
        if item not in unique:  # O(n) check each time
            unique.append(item)
    return unique


def slow_matrix_multiply(size: int = 50):
    """SLOW: Matrix multiplication with pure Python loops."""
    # Create two matrices
    a = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(i + j)
        a.append(row)

    b = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(i * j)
        b.append(row)

    # Multiply matrices (slow way)
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            total = 0
            for k in range(size):
                total = total + a[i][k] * b[k][j]
            row.append(total)
        result.append(row)

    return result


# Slow regex patterns
SLOW_REGEX_PATTERNS = [
    r"(a+)+b",
    r".*.*.*expensive",
    r"(\w+\s+)*\w+",
    r"([\w])+@([\w])+\.([\w])+",
    r"(\d{1}|\d{2}|\d{3})",
]


def run_function_optimization(func, kwargs, number, repeat):
    """Run optimization workflow for a single function."""
    print("\n" + "="*80)
    print(f"OPTIMIZING: {func.__name__}")
    print("="*80)

    # Get source code of slow function
    slow_source = inspect.getsource(func)

    print("\n[ORIGINAL SLOW CODE]")
    print("-" * 80)
    print(slow_source)
    print("-" * 80)

    # Benchmark slow version
    print("\n[BENCHMARKING SLOW VERSION...]")
    slow_results = benchmark_code(func, number=number, repeat=repeat, auto_generate_defaults=False, **kwargs)
    print_benchmark_results(f"{func.__name__} (SLOW)", slow_results)

    # Generate optimized code
    print("\n[GENERATING OPTIMIZED CODE...]")
    optimized_source = optimize(slow_source)

    print("\n[OPTIMIZED CODE]")
    print("-" * 80)
    print(optimized_source)
    print("-" * 80)

    # Execute the optimized code to create the function
    try:
        local_scope = {}
        exec(optimized_source, globals(), local_scope)
        optimized_func_name = func.__name__
        optimized_func = local_scope[optimized_func_name]

        # Benchmark optimized version
        print("\n[BENCHMARKING OPTIMIZED VERSION...]")
        optimized_results = benchmark_code(
            optimized_func,
            number=number,
            repeat=repeat,
            auto_generate_defaults=False,
            **kwargs
        )
        print_benchmark_results(f"{func.__name__} (OPTIMIZED)", optimized_results)

        # Calculate speedup
        speedup = slow_results['mean'] / optimized_results['mean']
        time_saved = slow_results['mean'] - optimized_results['mean']

        print("\n" + "🚀" * 40)
        print(f"PERFORMANCE IMPROVEMENT")
        print("🚀" * 40)
        print(f"Original time:   {slow_results['mean']:.6f} seconds")
        print(f"Optimized time:  {optimized_results['mean']:.6f} seconds")
        print(f"Time saved:      {time_saved:.6f} seconds ({time_saved/slow_results['mean']*100:.1f}%)")
        print(f"Speedup:         {speedup:.2f}x faster")
        print("🚀" * 40)

    except Exception as e:
        print(f"\n❌ Error executing optimized code: {e}")
        print("Skipping benchmark for optimized version...")


def run_function_benchmark(func, kwargs, number, repeat, name):
    """Run benchmark only (no optimization) for a function."""
    print(f"\n### {name} ###")
    results = benchmark_code(func, number=number, repeat=repeat, auto_generate_defaults=False, **kwargs)
    print_benchmark_results(name, results)


def run_regex_optimization(slow_pattern):
    """Run optimization workflow for a single regex pattern."""
    print("\n" + "="*80)
    print("OPTIMIZING REGEX")
    print("="*80)

    print("\n[ORIGINAL PATTERN]")
    print(f"Pattern: {slow_pattern}")

    # Generate test cases: 25 positive, 25 negative
    print("\n[GENERATING TEST CASES...]")
    positive_cases, negative_cases = generate_regex_test_cases(slow_pattern, num_positive=25, num_negative=25)
    test_data = positive_cases + negative_cases

    print(f"Generated {len(positive_cases)} positive cases (should match)")
    print(f"Generated {len(negative_cases)} negative cases (should NOT match)")
    print(f"Sample positive: {positive_cases[:3]}...")
    print(f"Sample negative: {negative_cases[:3]}...")

    # Benchmark slow pattern
    print("\n[BENCHMARKING ORIGINAL PATTERN...]")
    try:
        slow_stats, slow_results = benchmark_regex(slow_pattern, test_data, number=100, repeat=5)
        print_benchmark_results(f"Regex: {slow_pattern}", slow_stats)

        # Count matches for positive and negative cases
        slow_positive_matches = sum(slow_results[:len(positive_cases)])
        slow_negative_matches = sum(slow_results[len(positive_cases):])

        print(f"\nOriginal Pattern Results:")
        print(f"  Positive cases: {slow_positive_matches}/{len(positive_cases)} matched")
        print(f"  Negative cases: {slow_negative_matches}/{len(negative_cases)} matched (should be 0)")

        # Optimize pattern
        print("\n[OPTIMIZING PATTERN...]")
        optimized_pattern = optimize(slow_pattern)

        print(f"\n[OPTIMIZED PATTERN]")
        print(f"Pattern: {optimized_pattern}")

        # Benchmark optimized pattern
        print("\n[BENCHMARKING OPTIMIZED PATTERN...]")
        optimized_stats, optimized_results = benchmark_regex(optimized_pattern, test_data, number=100, repeat=5)
        print_benchmark_results(f"Regex: {optimized_pattern}", optimized_stats)

        # Count matches for positive and negative cases
        opt_positive_matches = sum(optimized_results[:len(positive_cases)])
        opt_negative_matches = sum(optimized_results[len(positive_cases):])

        print(f"\nOptimized Pattern Results:")
        print(f"  Positive cases: {opt_positive_matches}/{len(positive_cases)} matched")
        print(f"  Negative cases: {opt_negative_matches}/{len(negative_cases)} matched (should be 0)")

        # Validate results match
        print("\n" + "="*80)
        print("VALIDATION")
        print("="*80)

        if slow_results == optimized_results:
            print("✅ PERFECT MATCH: Both patterns produce identical results on all test cases")
        else:
            print("⚠️  WARNING: Patterns produce different results!")
            diff_count = sum(1 for a, b in zip(slow_results, optimized_results) if a != b)
            print(f"   Total differences: {diff_count}/{len(test_data)} strings")

            # Show differences in detail
            positive_diffs = sum(1 for i in range(len(positive_cases)) if slow_results[i] != optimized_results[i])
            negative_diffs = sum(1 for i in range(len(positive_cases), len(test_data)) if slow_results[i] != optimized_results[i])

            if positive_diffs > 0:
                print(f"   Positive cases with differences: {positive_diffs}/{len(positive_cases)}")
            if negative_diffs > 0:
                print(f"   Negative cases with differences: {negative_diffs}/{len(negative_cases)}")

        # Calculate speedup
        speedup = slow_stats['mean'] / optimized_stats['mean']
        time_saved = slow_stats['mean'] - optimized_stats['mean']

        print("\n" + "⚡" * 40)
        print("REGEX PERFORMANCE IMPROVEMENT")
        print("⚡" * 40)
        print(f"Original time:   {slow_stats['mean']:.6f} seconds")
        print(f"Optimized time:  {optimized_stats['mean']:.6f} seconds")
        print(f"Time saved:      {time_saved:.6f} seconds ({time_saved/slow_stats['mean']*100:.1f}%)")
        print(f"Speedup:         {speedup:.2f}x faster")
        print("⚡" * 40)

    except Exception as e:
        print(f"\n❌ Error processing regex: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("AI-POWERED CODE OPTIMIZATION TOOL")
    print("="*60)

    # Define all available examples
    function_optimizations = {
        "slow_nested_loops": (slow_nested_loops, {"n": 100}, 1000, 5),
        "slow_string_concat": (slow_string_concat, {"n": 500}, 100, 5),
        "slow_list_search": (slow_list_search, {"n": 1000}, 100, 5),
        "slow_filtering": (slow_filtering, {"n": 1000}, 1000, 5),
    }

    function_benchmarks = {
        "slow_sum_calculation": (slow_sum_calculation, {"n": 1000}, 1000, 5, "Slow Sum Calculation"),
        "slow_unique_values": (slow_unique_values, {"n": 500}, 100, 5, "Slow Unique Values"),
        "slow_dict_building": (slow_dict_building, {"n": 1000}, 1000, 5, "Slow Dictionary Building"),
        "slow_matrix_multiply": (slow_matrix_multiply, {"size": 50}, 10, 3, "Slow Matrix Multiplication"),
    }

    # Create menu choices
    choices = []

    # Add section header as disabled choice
    choices.append("──── FUNCTION OPTIMIZATIONS (with AI) ────")
    for key in function_optimizations.keys():
        choices.append(f"  Optimize: {key}")

    # Add function benchmarks
    choices.append("──── FUNCTION BENCHMARKS (benchmark only) ────")
    for key, (func, kwargs, number, repeat, name) in function_benchmarks.items():
        choices.append(f"  Benchmark: {key}")

    # Add regex patterns
    choices.append("──── REGEX OPTIMIZATIONS ────")
    for i, pattern in enumerate(SLOW_REGEX_PATTERNS):
        choices.append(f"  Regex: {pattern[:50]}")

    # Add special options
    choices.append("──── RUN MULTIPLE ────")
    choices.append("  Run All Function Optimizations")
    choices.append("  Run All Function Benchmarks")
    choices.append("  Run All Regex Optimizations")
    choices.append("  Run Everything")
    choices.append("────────────────────")
    choices.append("  Exit")

    # Show menu
    questions = [
        InquirerList('choice',
                     message="Select example to run (use arrow keys)",
                     choices=choices,
                     carousel=True)
    ]

    try:
        answer = prompt(questions, theme=GreenPassion())

        if not answer or answer['choice'].strip() == 'Exit':
            print("\nExiting...")
        else:
            choice = answer['choice'].strip()

            # Skip if it's a separator/header
            if choice.startswith('────'):
                print("\nPlease select an actual option, not a separator.")
            # Handle function optimizations
            elif 'Optimize:' in choice:
                func_name = choice.split('Optimize:')[1].strip()
                func, kwargs, number, repeat = function_optimizations[func_name]
                run_function_optimization(func, kwargs, number, repeat)

            # Handle function benchmarks
            elif 'Benchmark:' in choice:
                func_name = choice.split('Benchmark:')[1].strip()
                func, kwargs, number, repeat, name = function_benchmarks[func_name]
                run_function_benchmark(func, kwargs, number, repeat, name)

            # Handle regex optimizations
            elif 'Regex:' in choice:
                pattern_start = choice.split('Regex:')[1].strip()
                # Find matching regex pattern
                for pattern in SLOW_REGEX_PATTERNS:
                    if pattern.startswith(pattern_start):
                        run_regex_optimization(pattern)
                        break

            # Handle "run all" options
            elif 'Run All Function Optimizations' in choice:
                for func, kwargs, number, repeat in function_optimizations.values():
                    run_function_optimization(func, kwargs, number, repeat)

            elif 'Run All Function Benchmarks' in choice:
                print("\n\n" + "="*80)
                print("ADDITIONAL SLOW CODE EXAMPLES")
                print("="*80)
                for func, kwargs, number, repeat, name in function_benchmarks.values():
                    run_function_benchmark(func, kwargs, number, repeat, name)

            elif 'Run All Regex Optimizations' in choice:
                print("\n\n" + "="*80)
                print("REGEX PATTERN OPTIMIZATION")
                print("="*80)
                for pattern in SLOW_REGEX_PATTERNS:
                    run_regex_optimization(pattern)

            elif 'Run Everything' in choice:
                # Run everything
                for func, kwargs, number, repeat in function_optimizations.values():
                    run_function_optimization(func, kwargs, number, repeat)

                print("\n\n" + "="*80)
                print("ADDITIONAL SLOW CODE EXAMPLES")
                print("="*80)
                for func, kwargs, number, repeat, name in function_benchmarks.values():
                    run_function_benchmark(func, kwargs, number, repeat, name)

                print("\n\n" + "="*80)
                print("REGEX PATTERN OPTIMIZATION")
                print("="*80)
                for pattern in SLOW_REGEX_PATTERNS:
                    run_regex_optimization(pattern)

            print("\n" + "="*80)
            print("COMPLETE")
            print("="*80 + "\n")

    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
