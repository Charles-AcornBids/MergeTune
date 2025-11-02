"""
Test suite for the optimize module with slow example functions and benchmarks.
Run this file to see optimization examples.
"""

import inspect
from inquirer import List as InquirerList, prompt
from inquirer.themes import GreenPassion

from optimize import (
    benchmark_code,
    print_benchmark_results,
    optimize,
    generate_regex_test_cases,
    benchmark_regex,
    optimize_sql,
    analyze_sql_complexity,
)


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


# Slow SQL queries (INTENTIONALLY INEFFICIENT)
SLOW_SQL_QUERIES = [
    # Query 1: SELECT *, no WHERE, multiple JOINs
    """
    SELECT *
    FROM users
    JOIN orders ON users.id = orders.user_id
    JOIN order_items ON orders.id = order_items.order_id
    JOIN products ON order_items.product_id = products.id
    """,

    # Query 2: Subquery in SELECT
    """
    SELECT
        users.name,
        (SELECT COUNT(*) FROM orders WHERE orders.user_id = users.id) as order_count,
        (SELECT SUM(total) FROM orders WHERE orders.user_id = users.id) as total_spent
    FROM users
    """,

    # Query 3: Multiple OR conditions
    """
    SELECT * FROM products
    WHERE category = 'electronics' OR category = 'computers' OR category = 'phones' OR category = 'tablets' OR category = 'accessories'
    """,

    # Query 4: LIKE with leading wildcard
    """
    SELECT * FROM customers
    WHERE email LIKE '%@gmail.com'
    ORDER BY created_at DESC
    """,

    # Query 5: NOT IN with subquery
    """
    SELECT * FROM users
    WHERE id NOT IN (SELECT user_id FROM active_sessions)
    """,

    # Query 6: Complex subqueries
    """
    SELECT *
    FROM products
    WHERE price > (SELECT AVG(price) FROM products WHERE category = (SELECT category FROM products WHERE id = 123))
    """,
]


# Functions with SQL queries embedded (INTENTIONALLY SLOW)
def slow_user_query():
    """SLOW: Function with inefficient SQL - SELECT *, no WHERE clause."""
    query = """
    SELECT *
    FROM users
    JOIN orders ON users.id = orders.user_id
    """
    # In real code, this would execute the query
    return query


def slow_report_with_subqueries():
    """SLOW: Function with multiple correlated subqueries."""
    query = """
    SELECT
        users.id,
        users.name,
        (SELECT COUNT(*) FROM orders WHERE orders.user_id = users.id) as order_count,
        (SELECT COUNT(*) FROM reviews WHERE reviews.user_id = users.id) as review_count,
        (SELECT AVG(rating) FROM reviews WHERE reviews.user_id = users.id) as avg_rating
    FROM users
    WHERE users.status = 'active'
    """
    # Process the results
    return query


def slow_product_search(search_term: str = "laptop"):
    """SLOW: Function with LIKE leading wildcard and SELECT *."""
    query = f"""
    SELECT *
    FROM products
    WHERE name LIKE '%{search_term}%' OR description LIKE '%{search_term}%'
    ORDER BY created_at DESC
    """
    return query


def slow_inventory_check():
    """SLOW: Function with NOT IN and multiple operations."""
    query = """
    SELECT *
    FROM products
    WHERE id NOT IN (
        SELECT product_id FROM inventory WHERE quantity > 0
    ) OR category NOT IN (
        SELECT category FROM active_categories
    )
    """
    return query


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


def run_sql_optimization(query):
    """Run optimization workflow for a single SQL query."""
    print("\n" + "="*80)
    print("OPTIMIZING SQL QUERY")
    print("="*80)

    print("\n[ORIGINAL QUERY]")
    print("-" * 80)
    print(query.strip())
    print("-" * 80)

    # Analyze original query complexity
    print("\n[ANALYZING ORIGINAL QUERY...]")
    original_analysis = analyze_sql_complexity(query)

    print(f"\nComplexity Score: {original_analysis['score']}")
    print(f"\nOperations:")
    for op, count in original_analysis['operations'].items():
        print(f"  {op}: {count}")

    print(f"\nIssues Detected:")
    for issue in original_analysis['issues']:
        print(f"  - {issue}")

    print(f"\nQuery Details:")
    for key, value in original_analysis['details'].items():
        print(f"  {key}: {value}")

    # Optimize query
    print("\n[OPTIMIZING QUERY...]")
    result = optimize_sql(query)

    print("\n[OPTIMIZED QUERY]")
    print("-" * 80)
    print(result['optimized'].strip())
    print("-" * 80)

    # Analyze optimized query
    print("\n[ANALYZING OPTIMIZED QUERY...]")
    optimized_analysis = result['optimized_complexity']

    print(f"\nComplexity Score: {optimized_analysis['score']} (was {original_analysis['score']})")
    print(f"\nOperations:")
    for op, count in optimized_analysis['operations'].items():
        orig_count = original_analysis['operations'].get(op, 0)
        change = "" if count == orig_count else f" (was {orig_count})"
        print(f"  {op}: {count}{change}")

    print(f"\nIssues Detected:")
    for issue in optimized_analysis['issues']:
        print(f"  - {issue}")

    # Show improvements
    print("\n" + "⚡" * 40)
    print("SQL OPTIMIZATION RESULTS")
    print("⚡" * 40)

    print(f"\nImprovements Made:")
    for improvement in result['improvements']:
        print(f"  ✓ {improvement}")

    print(f"\nComplexity Reduction: {result['estimated_improvement']:.1f}%")

    if original_analysis['score'] > optimized_analysis['score']:
        print(f"Score improved from {original_analysis['score']} to {optimized_analysis['score']}")
    elif original_analysis['score'] < optimized_analysis['score']:
        print(f"⚠️  Note: Optimized query has higher complexity score")
    else:
        print(f"Complexity score unchanged (both {original_analysis['score']})")

    print("⚡" * 40)


def run_function_with_sql_optimization(func):
    """Run optimization for a function containing SQL queries."""
    print("\n" + "="*80)
    print(f"OPTIMIZING FUNCTION WITH SQL: {func.__name__}")
    print("="*80)

    # Get source code
    func_source = inspect.getsource(func)

    print("\n[ORIGINAL FUNCTION]")
    print("-" * 80)
    print(func_source)
    print("-" * 80)

    # Optimize the entire function (AI will detect SQL and optimize it)
    print("\n[OPTIMIZING FUNCTION...]")
    optimized_source = optimize(func_source)

    print("\n[OPTIMIZED FUNCTION]")
    print("-" * 80)
    print(optimized_source)
    print("-" * 80)

    print("\n" + "✓" * 40)
    print("OPTIMIZATION COMPLETE")
    print("✓" * 40)


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

    # SQL optimization examples
    sql_functions = {
        "slow_user_query": slow_user_query,
        "slow_report_with_subqueries": slow_report_with_subqueries,
        "slow_product_search": slow_product_search,
        "slow_inventory_check": slow_inventory_check,
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

    # Add SQL optimizations
    choices.append("──── SQL QUERY OPTIMIZATIONS ────")
    for i, query in enumerate(SLOW_SQL_QUERIES):
        query_preview = query.strip().replace('\n', ' ')[:60] + "..."
        choices.append(f"  SQL Query {i+1}: {query_preview}")

    # Add SQL functions
    choices.append("──── SQL FUNCTIONS (SQL in Python) ────")
    for key in sql_functions.keys():
        choices.append(f"  SQL Function: {key}")

    # Add regex patterns
    choices.append("──── REGEX OPTIMIZATIONS ────")
    for i, pattern in enumerate(SLOW_REGEX_PATTERNS):
        choices.append(f"  Regex: {pattern[:50]}")

    # Add special options
    choices.append("──── RUN MULTIPLE ────")
    choices.append("  Run All Function Optimizations")
    choices.append("  Run All Function Benchmarks")
    choices.append("  Run All SQL Queries")
    choices.append("  Run All SQL Functions")
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

            # Handle SQL query optimizations
            elif 'SQL Query' in choice:
                # Extract query number
                query_num = int(choice.split('SQL Query')[1].split(':')[0].strip()) - 1
                if 0 <= query_num < len(SLOW_SQL_QUERIES):
                    run_sql_optimization(SLOW_SQL_QUERIES[query_num])

            # Handle SQL function optimizations
            elif 'SQL Function:' in choice:
                func_name = choice.split('SQL Function:')[1].strip()
                if func_name in sql_functions:
                    run_function_with_sql_optimization(sql_functions[func_name])

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

            elif 'Run All SQL Queries' in choice:
                print("\n\n" + "="*80)
                print("SQL QUERY OPTIMIZATION")
                print("="*80)
                for query in SLOW_SQL_QUERIES:
                    run_sql_optimization(query)

            elif 'Run All SQL Functions' in choice:
                print("\n\n" + "="*80)
                print("SQL FUNCTION OPTIMIZATION")
                print("="*80)
                for func in sql_functions.values():
                    run_function_with_sql_optimization(func)

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
                print("SQL QUERY OPTIMIZATION")
                print("="*80)
                for query in SLOW_SQL_QUERIES:
                    run_sql_optimization(query)

                print("\n\n" + "="*80)
                print("SQL FUNCTION OPTIMIZATION")
                print("="*80)
                for func in sql_functions.values():
                    run_function_with_sql_optimization(func)

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
