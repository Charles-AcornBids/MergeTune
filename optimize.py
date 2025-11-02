import timeit
import statistics
import inspect
import os
import re
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where, Function, Comparison
from sqlparse.tokens import Keyword, DML
from typing import Callable, Any, Dict, List, Tuple, Optional
from dotenv import load_dotenv
from openai import OpenAI

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
    # Remove whitespace for analysis
    stripped = text.strip()

    # Python code keywords that indicate it's NOT a regex
    python_keywords = ['def ', 'class ', 'import ', 'from ', 'return ', 'if ', 'for ', 'while ', 'try:', 'except']
    if any(keyword in stripped for keyword in python_keywords):
        return False

    # If it has multiple lines with significant content, likely not a regex
    lines = [line.strip() for line in stripped.split('\n') if line.strip()]
    if len(lines) > 3:
        return False

    # If it's short and contains regex metacharacters, likely a pattern
    if len(stripped) < 200 and any(char in stripped for char in ['+', '*', '?', '|', '[', ']', '{', '}', '^', '$']):
        # Try to compile it as regex
        try:
            re.compile(stripped)
            return True
        except Exception:
            pass

    return False


def _is_sql_query(text: str) -> bool:
    """
    Detect if a string is likely a SQL query.
    """
    # Strip whitespace and convert to uppercase for checking
    text_upper = text.strip().upper()

    # Check for SQL keywords at the start
    sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP', 'WITH']
    if any(text_upper.startswith(keyword) for keyword in sql_keywords):
        return True

    # Try to parse as SQL
    try:
        parsed = sqlparse.parse(text)
        if parsed and len(parsed) > 0:
            # Check if it has SQL statement types
            for statement in parsed:
                if statement.get_type() in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP', 'UNKNOWN']:
                    # UNKNOWN might be valid SQL too, so check for FROM, WHERE, JOIN
                    tokens_str = str(statement).upper()
                    if any(kw in tokens_str for kw in ['FROM', 'WHERE', 'JOIN', 'GROUP BY', 'ORDER BY', 'SET', 'VALUES']):
                        return True
                    if statement.get_type() != 'UNKNOWN':
                        return True
    except Exception:
        pass

    return False


def _extract_sql_from_code(code: str) -> List[Tuple[str, int, int]]:
    """
    Extract SQL queries from Python code (strings, multiline strings, etc.).

    Returns:
        List of tuples (sql_query, start_line, end_line)
    """
    sql_queries = []

    # Pattern to find SQL in strings (both single and triple quoted)
    # Look for strings that contain SQL keywords
    lines = code.split('\n')
    current_sql = []
    in_sql_string = False
    start_line = 0
    quote_type = None

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Check for start of multiline string with SQL
        if not in_sql_string:
            for qt in ['"""', "'''", '"', "'"]:
                if qt in stripped:
                    # Extract the string content
                    potential_sql = stripped
                    # Remove common prefixes (f-string, raw string, etc.)
                    potential_sql = re.sub(r'^[frFR]*["\']', '', potential_sql)

                    if any(kw in potential_sql.upper() for kw in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'FROM', 'WHERE']):
                        in_sql_string = True
                        start_line = i
                        quote_type = qt
                        current_sql.append(line)

                        # Check if it's a single-line string
                        if stripped.count(qt) >= 2:
                            in_sql_string = False
                            sql_text = '\n'.join(current_sql)
                            # Try to extract just the SQL part
                            extracted = _extract_sql_string_content(sql_text)
                            if extracted and _is_sql_query(extracted):
                                sql_queries.append((extracted, start_line, i))
                            current_sql = []
                        break
        else:
            current_sql.append(line)
            if quote_type in line:
                in_sql_string = False
                sql_text = '\n'.join(current_sql)
                extracted = _extract_sql_string_content(sql_text)
                if extracted and _is_sql_query(extracted):
                    sql_queries.append((extracted, start_line, i))
                current_sql = []

    return sql_queries


def _extract_sql_string_content(text: str) -> Optional[str]:
    """
    Extract SQL content from a Python string assignment.
    Handles f-strings, raw strings, and regular strings.
    """
    # Remove common Python string prefixes and quotes
    patterns = [
        r'[frFR]*"""(.+?)"""',
        r"[frFR]*'''(.+?)'''",
        r'[frFR]*"(.+?)"',
        r"[frFR]*'(.+?)'",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    # If no match, try to clean it up manually
    text = re.sub(r'^[^"\']*["\']', '', text)
    text = re.sub(r'["\'][^"\']*$', '', text)
    return text.strip()


def analyze_sql_complexity(query: str) -> Dict[str, Any]:
    """
    Analyze SQL query complexity without executing it.

    Args:
        query: SQL query string

    Returns:
        Dictionary with complexity metrics and issues:
        - score: Complexity score (higher = more complex/problematic)
        - issues: List of detected anti-patterns
        - operations: Dict of operation counts
        - details: Additional details about the query
    """
    parsed = sqlparse.parse(query)
    if not parsed:
        return {
            "score": 0,
            "issues": ["Unable to parse SQL query"],
            "operations": {},
            "details": {}
        }

    statement = parsed[0]
    issues = []
    operations = {
        "joins": 0,
        "subqueries": 0,
        "wildcards": 0,
        "functions": 0,
        "or_conditions": 0,
        "tables": 0,
    }

    query_upper = query.upper()

    # Check for SELECT *
    if re.search(r'SELECT\s+\*', query, re.IGNORECASE):
        issues.append("Using SELECT * (specify columns instead)")
        operations["wildcards"] += 1

    # Count JOINs
    join_count = len(re.findall(r'\bJOIN\b', query_upper))
    operations["joins"] = join_count
    if join_count > 3:
        issues.append(f"Many JOINs ({join_count}) - consider denormalization or restructuring")

    # Count subqueries
    subquery_count = query.count('(SELECT') + query.count('( SELECT')
    operations["subqueries"] = subquery_count
    if subquery_count > 2:
        issues.append(f"Multiple subqueries ({subquery_count}) - consider CTEs or JOINs")

    # Check for OR conditions (can be slow)
    or_count = len(re.findall(r'\bOR\b', query_upper))
    operations["or_conditions"] = or_count
    if or_count > 3:
        issues.append(f"Many OR conditions ({or_count}) - consider IN clause or UNION")

    # Check for functions that might be expensive
    expensive_functions = ['DISTINCT', 'GROUP BY', 'ORDER BY', 'HAVING']
    for func in expensive_functions:
        if func in query_upper:
            operations["functions"] += 1

    # Check for missing WHERE clause in SELECT
    if query_upper.strip().startswith('SELECT') and 'WHERE' not in query_upper:
        if 'LIMIT' not in query_upper:
            issues.append("No WHERE clause - will scan entire table")

    # Check for LIKE with leading wildcard
    if re.search(r"LIKE\s+['\"]%", query, re.IGNORECASE):
        issues.append("LIKE with leading wildcard (%) prevents index usage")

    # Check for NOT IN (can be slow)
    if 'NOT IN' in query_upper:
        issues.append("NOT IN clause - consider LEFT JOIN with NULL check instead")

    # Count tables (approximate)
    from_match = re.search(r'FROM\s+([\w\s,]+?)(?:WHERE|JOIN|GROUP|ORDER|LIMIT|;|$)', query, re.IGNORECASE)
    if from_match:
        tables = [t.strip() for t in from_match.group(1).split(',')]
        operations["tables"] = len(tables)

    # Calculate complexity score
    score = (
        operations["joins"] * 10 +
        operations["subqueries"] * 15 +
        operations["wildcards"] * 5 +
        operations["functions"] * 3 +
        operations["or_conditions"] * 2 +
        len(issues) * 8
    )

    # Estimate query characteristics
    details = {
        "query_type": statement.get_type(),
        "has_where": "WHERE" in query_upper,
        "has_index_hints": "INDEX" in query_upper or "USE INDEX" in query_upper,
        "has_limit": "LIMIT" in query_upper,
        "estimated_rows_scanned": "unknown" if "WHERE" in query_upper else "all rows"
    }

    return {
        "score": score,
        "issues": issues if issues else ["No major issues detected"],
        "operations": operations,
        "details": details
    }


def optimize_sql(query: str, schema: str = "", context: str = "") -> Dict[str, Any]:
    """
    Optimize SQL query using GPT-4o with static analysis.

    Args:
        query: SQL query to optimize
        schema: Optional database schema information
        context: Optional context about the use case

    Returns:
        Dictionary with:
        - original: Original query
        - optimized: Optimized query
        - improvements: List of improvements made
        - original_complexity: Complexity analysis of original
        - optimized_complexity: Complexity analysis of optimized
        - estimated_improvement: Estimated improvement percentage
    """
    # Analyze original query
    original_complexity = analyze_sql_complexity(query)

    # Build prompt for GPT-4o
    prompt = f"""Optimize this SQL query for better performance:

```sql
{query}
```

"""

    if schema:
        prompt += f"\nDatabase Schema:\n{schema}\n"

    if context:
        prompt += f"\nContext: {context}\n"

    prompt += """
Focus on:
1. Index usage optimization
2. Reducing unnecessary operations
3. Rewriting subqueries as JOINs where appropriate
4. Using CTEs for better readability
5. Avoiding SELECT *
6. Removing redundant conditions

Return ONLY the optimized SQL query, nothing else. No explanations, no markdown formatting."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a SQL optimization expert. Return only optimized SQL queries without any explanation or formatting."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )

        optimized_query = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if optimized_query.startswith("```"):
            lines = optimized_query.split("\n")
            optimized_query = "\n".join([line for line in lines if not line.startswith("```")])
            optimized_query = optimized_query.strip()

        # Remove language identifier if present
        if optimized_query.startswith("sql\n"):
            optimized_query = optimized_query[4:]

        # Analyze optimized query
        optimized_complexity = analyze_sql_complexity(optimized_query)

        # Calculate improvement
        if original_complexity["score"] > 0:
            improvement_pct = ((original_complexity["score"] - optimized_complexity["score"]) / original_complexity["score"]) * 100
        else:
            improvement_pct = 0

        # Determine improvements made
        improvements = []
        if original_complexity["operations"]["wildcards"] > optimized_complexity["operations"]["wildcards"]:
            improvements.append("Replaced SELECT * with specific columns")
        if original_complexity["operations"]["subqueries"] > optimized_complexity["operations"]["subqueries"]:
            improvements.append("Reduced subquery usage")
        if original_complexity["operations"]["joins"] < optimized_complexity["operations"]["joins"] and original_complexity["operations"]["subqueries"] > optimized_complexity["operations"]["subqueries"]:
            improvements.append("Converted subqueries to JOINs")
        if "No WHERE clause" in original_complexity["issues"] and "No WHERE clause" not in optimized_complexity["issues"]:
            improvements.append("Added WHERE clause")
        if len(original_complexity["issues"]) > len(optimized_complexity["issues"]):
            improvements.append(f"Reduced issues from {len(original_complexity['issues'])} to {len(optimized_complexity['issues'])}")

        if not improvements:
            improvements.append("Query structure refined for better clarity")

        return {
            "original": query,
            "optimized": optimized_query,
            "improvements": improvements,
            "original_complexity": original_complexity,
            "optimized_complexity": optimized_complexity,
            "estimated_improvement": improvement_pct
        }

    except Exception as e:
        print(f"Error optimizing SQL: {e}")
        return {
            "original": query,
            "optimized": query,
            "improvements": ["Error occurred during optimization"],
            "original_complexity": original_complexity,
            "optimized_complexity": original_complexity,
            "estimated_improvement": 0
        }


def optimize(code: str, benchmark: bool = True) -> Dict[str, Any]:
    """
    Optimize code, SQL queries, or regex patterns.

    Args:
        code: The code, SQL query, or regex pattern to optimize
        benchmark: If True, run benchmarks and validation (default: True)

    Returns:
        Dictionary with:
        - type: Type of code (sql, regex, python)
        - original: Original code
        - optimized: Optimized code
        - original_benchmark: Benchmark results for original (if applicable)
        - optimized_benchmark: Benchmark results for optimized (if applicable)
        - speedup: Speedup factor (if benchmarked)
        - validation: Validation results (if applicable)
        - improvements: List of improvements made
    """
    # Detect input type: SQL, regex, or Python code
    is_sql = _is_sql_query(code)
    is_regex = _is_regex_pattern(code) if not is_sql else False

    if is_sql:
        # Use dedicated SQL optimizer (already returns full results)
        result = optimize_sql(code)
        return {
            "type": "sql",
            "original": result["original"],
            "optimized": result["optimized"],
            "original_benchmark": None,  # SQL doesn't execute, just static analysis
            "optimized_benchmark": None,
            "speedup": None,
            "validation": {
                "original_complexity": result["original_complexity"],
                "optimized_complexity": result["optimized_complexity"],
                "estimated_improvement": result["estimated_improvement"]
            },
            "improvements": result["improvements"]
        }

    elif is_regex:
        # Optimize regex pattern
        prompt = f"""Optimize this regex pattern:

{code}

Return only the optimized pattern, nothing else."""

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

            # Remove quotes from regex patterns
            optimized = optimized.strip('"\'`')

            # Benchmark if requested
            if benchmark:
                # Generate test cases
                positive_cases, negative_cases = generate_regex_test_cases(code, num_positive=25, num_negative=25)
                test_data = positive_cases + negative_cases

                # Benchmark original
                original_stats, original_results = benchmark_regex(code, test_data, number=100, repeat=5)

                # Benchmark optimized
                optimized_stats, optimized_results = benchmark_regex(optimized, test_data, number=100, repeat=5)

                # Calculate speedup
                speedup = original_stats['mean'] / optimized_stats['mean'] if optimized_stats['mean'] > 0 else 0

                # Validate results match
                results_match = original_results == optimized_results
                diff_count = sum(1 for a, b in zip(original_results, optimized_results) if a != b)

                original_positive_matches = sum(original_results[:len(positive_cases)])
                original_negative_matches = sum(original_results[len(positive_cases):])
                optimized_positive_matches = sum(optimized_results[:len(positive_cases)])
                optimized_negative_matches = sum(optimized_results[len(positive_cases):])

                return {
                    "type": "regex",
                    "original": code,
                    "optimized": optimized,
                    "original_benchmark": original_stats,
                    "optimized_benchmark": optimized_stats,
                    "speedup": speedup,
                    "validation": {
                        "results_match": results_match,
                        "differences": diff_count,
                        "total_tests": len(test_data),
                        "original_positive_matches": f"{original_positive_matches}/{len(positive_cases)}",
                        "original_negative_matches": f"{original_negative_matches}/{len(negative_cases)}",
                        "optimized_positive_matches": f"{optimized_positive_matches}/{len(positive_cases)}",
                        "optimized_negative_matches": f"{optimized_negative_matches}/{len(negative_cases)}",
                        "test_data": test_data,
                        "positive_cases_count": len(positive_cases),
                        "negative_cases_count": len(negative_cases)
                    },
                    "improvements": ["Pattern optimized for better performance"]
                }
            else:
                return {
                    "type": "regex",
                    "original": code,
                    "optimized": optimized,
                    "original_benchmark": None,
                    "optimized_benchmark": None,
                    "speedup": None,
                    "validation": None,
                    "improvements": ["Pattern optimized for better performance"]
                }

        except Exception as e:
            print(f"Error optimizing regex: {e}")
            return {
                "type": "regex",
                "original": code,
                "optimized": code,
                "original_benchmark": None,
                "optimized_benchmark": None,
                "speedup": 1.0,
                "validation": {"error": str(e)},
                "improvements": ["Error occurred during optimization"]
            }

    else:
        # Python code optimization (no automatic benchmarking - requires execution context)
        prompt = f"""Optimize this Python code:

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

            return {
                "type": "python",
                "original": code,
                "optimized": optimized,
                "original_benchmark": None,
                "optimized_benchmark": None,
                "speedup": None,
                "validation": {"note": "Python code requires manual benchmarking with execution context"},
                "improvements": ["Code optimized for better performance"]
            }

        except Exception as e:
            print(f"Error optimizing Python code: {e}")
            return {
                "type": "python",
                "original": code,
                "optimized": code,
                "original_benchmark": None,
                "optimized_benchmark": None,
                "speedup": None,
                "validation": {"error": str(e)},
                "improvements": ["Error occurred during optimization"]
            }


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
