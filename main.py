"""
MergeTune - AI-powered PR optimizer that suggests code optimizations
"""
import asyncio
import os
import json
import re
import ast
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any
import nivara as nv

from openai import AsyncOpenAI
from metorial import Metorial, MetorialOpenAI, MetorialAPIError

from optimize import optimize, _extract_sql_from_code, _is_regex_pattern
from openai import OpenAI

# Load environment variables
load_dotenv()

nv.configure(
    timeout=2.0,
    retries=2,
    mode='background',
    queue_size=10000,
    debug=True,
)

# OAuth cache file path
OAUTH_CACHE_FILE = Path(".oauth_cache.json")

# Repository configuration
GITHUB_REPO_OWNER = os.getenv("GITHUB_REPO_OWNER", "Charles-AcornBids")
GITHUB_REPO_NAME = os.getenv("GITHUB_REPO_NAME", "YC_Agent_Jam_Example")

# Initialize OpenAI client for optimization analysis
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_cached_oauth_session() -> dict | None:
    """Load cached OAuth session from file if it exists."""
    if OAUTH_CACHE_FILE.exists():
        try:
            with open(OAUTH_CACHE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  Warning: Could not load cached OAuth session: {e}")
            return None
    return None


def save_oauth_session(session_id: str, deployment_id: str) -> None:
    """Save OAuth session to cache file."""
    try:
        cache_data = {
            "session_id": session_id,
            "deployment_id": deployment_id
        }
        with open(OAUTH_CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print("💾 OAuth session cached successfully")
    except IOError as e:
        print(f"⚠️  Warning: Could not save OAuth session: {e}")


async def setup_oauth_if_needed(metorial: Metorial, deployment_id: str):
    """
    Set up OAuth authentication for deployments that require it (e.g., GitHub).
    Checks for cached OAuth session first, creates new one if needed.
    """
    # Check for cached OAuth session
    cached = load_cached_oauth_session()

    if cached and cached.get("deployment_id") == deployment_id:
        print("🔑 Found cached OAuth session, attempting to use it...")
        try:
            # Create a simple object with the cached session ID
            class CachedSession:
                def __init__(self, session_id):
                    self.id = session_id

            return CachedSession(cached["session_id"])
        except Exception as e:
            print(f"⚠️  Cached session failed: {e}")
            print("🔄 Creating new OAuth session...")

    # Create new OAuth session
    print("🔗 Creating OAuth session...")
    oauth_session = metorial.oauth.sessions.create(
        server_deployment_id=deployment_id
    )

    print(f"📋 Please authenticate at: {oauth_session.url}")

    print("\n⏳ Waiting for OAuth completion...")
    await metorial.oauth.wait_for_completion([oauth_session])

    print("✅ OAuth session completed!\n")

    # Cache the session
    save_oauth_session(oauth_session.id, deployment_id)

    return oauth_session


def identify_individual_speedups(code: str, file_path: str, start_line: int = 1) -> List[Dict[str, Any]]:
    """
    Identify individual speedup opportunities within a code block.

    Returns:
        List of dicts with 'code_snippet', 'line_start', 'line_end', 'issue', 'suggestion'
    """
    prompt = f"""Analyze this Python code and identify INDIVIDUAL performance optimization opportunities.

```python
{code}
```

For each optimization opportunity, provide:
1. The specific code snippet that needs optimization (extract just the relevant lines)
2. The line numbers where it appears - IMPORTANT: Line numbers must be 1-indexed relative to the code above, where line 1 is the FIRST line shown above
3. A brief description of the performance issue
4. A suggested optimization approach

CRITICAL: Line numbering example:
- If the first line of the code above needs optimization, use "line_start": 1
- If the second line needs optimization, use "line_start": 2
- If lines 5-7 need optimization, use "line_start": 5, "line_end": 7

Return ONLY a JSON array in this exact format (no markdown, no explanation):
[
  {{
    "code_snippet": "the exact code that needs optimization",
    "line_start": 5,
    "line_end": 7,
    "issue": "Using string concatenation in loop",
    "suggestion": "Use str.join() or list append with join"
  }},
  ...
]

Focus on common Python performance issues like:
- String concatenation in loops
- Nested loops that could be list comprehensions
- Repeated list.append() that could be list comprehension
- Using 'in' operator on lists instead of sets
- Inefficient dictionary operations
- Unnecessary loops
- Repeated function calls that could be cached

Only include optimizations that would have a meaningful performance impact.
If there are no significant optimization opportunities, return an empty array: []
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a Python performance optimization expert. Return only valid JSON arrays, no markdown formatting."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )

        result_str = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if result_str.startswith("```"):
            lines = result_str.split("\n")
            result_str = "\n".join(
                [line for line in lines if not line.startswith("```")])
            result_str = result_str.strip()

        # Remove json language identifier if present
        if result_str.startswith("json\n"):
            result_str = result_str[5:]

        speedups = json.loads(result_str)

        # Adjust line numbers to be absolute
        # The LLM returns 1-indexed line numbers relative to the code snippet
        # start_line is the 1-indexed line number in the file where the code starts
        # So: absolute_line = start_line + (relative_line - 1)
        for speedup in speedups:
            relative_start = speedup['line_start']
            relative_end = speedup['line_end']

            # Convert relative line numbers to absolute
            speedup['line_start'] = start_line + relative_start - 1
            speedup['line_end'] = start_line + relative_end - 1
            speedup['location'] = f"{file_path}:{speedup['line_start']}-{speedup['line_end']}"

            # Debug output
            print(f"      Line number mapping: relative {relative_start}-{relative_end} -> absolute {speedup['line_start']}-{speedup['line_end']}")

        return speedups

    except Exception as e:
        print(f"      ⚠️  Error identifying individual speedups: {e}")
        return []


def extract_functions_from_code(code: str) -> List[Dict[str, Any]]:
    """
    Extract function definitions from Python code.

    Returns:
        List of dicts with 'name', 'code', 'line_start', 'line_end'
    """
    functions = []

    try:
        tree = ast.parse(code)
        lines = code.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Get the function's source code
                # node.lineno and node.end_lineno are both 1-indexed
                start_line = node.lineno - 1  # Convert to 0-indexed for slicing
                end_line_absolute = node.end_lineno if hasattr(
                    node, 'end_lineno') else node.lineno

                # Extract function code (slicing is exclusive of end, so we use end_line_absolute directly)
                func_code = '\n'.join(lines[start_line:end_line_absolute])

                functions.append({
                    'name': node.name,
                    'code': func_code,
                    'line_start': node.lineno,  # 1-indexed, line where function starts
                    'line_end': end_line_absolute  # 1-indexed, line where function ends (inclusive)
                })
    except SyntaxError as e:
        print(f"⚠️  Syntax error parsing code: {e}")

    return functions


def extract_optimizable_code(file_path: str, content: str) -> List[Dict[str, Any]]:
    """
    Extract code segments that could be optimized from a Python file.

    Returns:
        List of dicts with 'type', 'code', 'location', 'description'
    """
    optimizable = []
    lines = content.split('\n')

    # Extract SQL queries
    sql_queries = _extract_sql_from_code(content)
    for sql, start_line, end_line in sql_queries:
        # Get the full line(s) of code for context (to preserve indentation and variable assignment)
        full_lines = '\n'.join(lines[start_line - 1:end_line])

        optimizable.append({
            'type': 'sql',
            'code': sql,
            'location': f"{file_path}:{start_line}-{end_line}",
            'description': f"SQL query at lines {start_line}-{end_line}",
            'full_lines': full_lines.rstrip(),
            'original_sql': sql
        })

    # Extract functions (we'll optimize the entire function)
    functions = extract_functions_from_code(content)
    for func in functions:
        # Skip very small functions (< 5 lines)
        if func['line_end'] - func['line_start'] > 5:
            optimizable.append({
                'type': 'python',
                'code': func['code'],
                'location': f"{file_path}:{func['line_start']}-{func['line_end']}",
                'description': f"Function '{func['name']}' at lines {func['line_start']}-{func['line_end']}",
                'line_start': func['line_start']
            })

    # Look for potential regex patterns in strings
    # Simple heuristic: find r"..." or r'...' strings
    regex_pattern = r"r['\"]([^'\"]+)['\"]"
    lines = content.split('\n')
    for match in re.finditer(regex_pattern, content):
        potential_regex = match.group(1)
        if _is_regex_pattern(potential_regex):
            # Debug: Let's see what we're matching and where
            match_start = match.start()
            newline_count = content[:match_start].count('\n')
            line_num = newline_count + 1

            # Get the full line of code for context
            full_line = lines[line_num - 1] if line_num <= len(lines) else ""

            # Debug output
            print(f"      DEBUG - Regex detection:")
            print(f"        Match start position: {match_start}")
            print(f"        Newlines before match: {newline_count}")
            print(f"        Calculated line number: {line_num}")
            print(f"        Full line content: {full_line[:100]}")
            print(f"        Total lines in file: {len(lines)}")

            optimizable.append({
                'type': 'regex',
                'code': potential_regex,
                'location': f"{file_path}:{line_num}",
                'description': f"Regex pattern at line {line_num}",
                # Only strip trailing whitespace, keep leading indentation
                'full_line': full_line.rstrip()
            })

    return optimizable


def reconstruct_code_with_optimization(original_code: str, original_pattern: str, optimized_pattern: str) -> str:
    """
    Use LLM to reconstruct complete code with an optimized pattern (regex, SQL, etc.),
    preserving variable names, quotes style, indentation, and other context.

    Args:
        original_code: The original line(s) of code
        original_pattern: The original pattern (regex, SQL query, etc.)
        optimized_pattern: The optimized pattern

    Returns:
        Complete reconstructed code with optimized pattern
    """
    prompt = f"""You are reconstructing Python code with an optimized pattern (could be SQL query, regex, or other pattern).

Original code:
{original_code}

Original pattern to replace:
{original_pattern}

Optimized pattern:
{optimized_pattern}

Replace ONLY the pattern/query in the original code with the optimized version, keeping everything else exactly the same:
- Preserve exact indentation/leading whitespace
- Preserve variable names
- Preserve quote style (single/double/triple quotes)
- Preserve any prefixes (f-string, r-string, etc.)
- Preserve any surrounding code structure

Return ONLY the complete reconstructed code line(s), nothing else. No explanations, no markdown."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You reconstruct code by replacing patterns (SQL, regex, etc.) while preserving all other context including indentation, variable names, and code structure. Return only the code, no explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )

        reconstructed = response.choices[0].message.content.rstrip()

        # Remove markdown code blocks if present
        if reconstructed.startswith("```"):
            lines = reconstructed.split("\n")
            reconstructed = "\n".join(
                [line for line in lines if not line.startswith("```")])
            reconstructed = reconstructed.lstrip("\n").rstrip()

        # Remove python language identifier if present
        if reconstructed.startswith("python\n"):
            reconstructed = reconstructed[7:]

        return reconstructed

    except Exception as e:
        print(f"      ⚠️  Error reconstructing code: {e}")
        # Fallback: try simple string replacement
        return original_code.replace(original_pattern, optimized_pattern)


def format_code_suggestion(opt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a single optimization as a GitHub code suggestion review comment.

    Args:
        opt: Single optimization result

    Returns:
        Dict with 'path', 'line', 'body' for GitHub review comment
    """
    result = opt['result']
    location = opt['location']
    description = opt['description']
    opt_type = opt.get('type', result.get('type', 'unknown'))

    # Parse location (format: "file_path:line_start-line_end" or "file_path:line_num")
    # Try multi-line format first
    location_match = re.match(r'([^:]+):(\d+)-(\d+)', location)
    if location_match:
        file_path = location_match.group(1)
        line_start = int(location_match.group(2))
        line_end = int(location_match.group(3))
    else:
        # Try single-line format
        location_match = re.match(r'([^:]+):(\d+)', location)
        if not location_match:
            return None
        file_path = location_match.group(1)
        line_start = int(location_match.group(2))
        line_end = line_start

    # Build the comment body with GitHub code suggestion syntax
    comment = f"**🚀 MergeTune Optimization Suggestion**\n\n"
    comment += f"**Issue:** {description}\n\n"

    # Add type-specific details
    if opt_type == 'speedup':
        if opt.get('suggestion'):
            comment += f"**Suggestion:** {opt['suggestion']}\n\n"

        if result.get('improvements'):
            comment += "**Improvements:**\n\n"
            for imp in result['improvements']:
                comment += f"- {imp}\n"
            comment += "\n"

        # Add the code suggestion
        comment += "```suggestion\n"
        comment += result['optimized'].rstrip() + "\n"
        comment += "```\n"

    elif opt_type == 'sql' or result['type'] == 'sql':
        if result.get('validation'):
            val = result['validation']
            orig_score = val['original_complexity']['score']
            opt_score = val['optimized_complexity']['score']
            improvement = val['estimated_improvement']
            comment += f"**Complexity Score:** {orig_score} → {opt_score} ({improvement:.1f}% improvement)\n\n"

        if result.get('improvements'):
            comment += "**Improvements:**\n\n"
            for imp in result['improvements']:
                comment += f"- {imp}\n"
            comment += "\n"

        # For SQL, reconstruct the full line(s) with the optimized SQL query
        full_lines = opt.get('full_lines', '')
        original_sql = opt.get('original_sql', result['original'])
        optimized_sql = opt.get('optimized_sql', result['optimized'])

        if full_lines:
            # Use LLM to reconstruct complete code with optimized SQL
            print(f"      Reconstructing SQL code with optimized query...")
            reconstructed_code = reconstruct_code_with_optimization(
                full_lines,
                original_sql,
                optimized_sql
            )
            comment += "```suggestion\n"
            comment += reconstructed_code.rstrip() + "\n"
            comment += "```\n"
        else:
            # Fallback: suggest just the SQL
            comment += "```suggestion\n"
            comment += result['optimized'].rstrip() + "\n"
            comment += "```\n"

    elif opt_type == 'regex' or result['type'] == 'regex':
        if result.get('speedup') and result['speedup'] > 1:
            comment += f"**Performance:** {result['speedup']:.2f}x faster\n\n"

        if result.get('improvements'):
            comment += "**Improvements:**\n"
            for imp in result['improvements']:
                comment += f"- {imp}\n"
            comment += "\n"

        # For regex, reconstruct the full line with the optimized pattern
        full_line = opt.get('full_line', '')
        original_pattern = opt.get('original_pattern', result['original'])
        optimized_pattern = opt.get('optimized_pattern', result['optimized'])

        if full_line:
            # Use LLM to reconstruct complete code with optimized pattern
            print(f"      Reconstructing code with optimized pattern...")
            reconstructed_code = reconstruct_code_with_optimization(
                full_line,
                original_pattern,
                optimized_pattern
            )
            comment += "```suggestion\n"
            comment += reconstructed_code.rstrip() + "\n"
            comment += "```\n"
        else:
            # Fallback: suggest just the pattern
            comment += "```suggestion\n"
            comment += result['optimized'].rstrip() + "\n"
            comment += "```\n"

    elif result['type'] == 'python':
        if result.get('improvements'):
            comment += "**Improvements:**\n\n"
            for imp in result['improvements']:
                comment += f"- {imp}\n"
            comment += "\n"

        # Add the code suggestion
        comment += "```suggestion\n"
        comment += result['optimized'].rstrip() + "\n"
        comment += "```\n"

    comment += "\n*Generated by MergeTune - AI-powered code optimization*"

    # Debug output
    print(f"      GitHub suggestion will be posted at {file_path}:{line_start}")

    return {
        'path': file_path,
        'line': line_start,
        'body': comment
    }


async def session_action(session):
    """Main action to analyze PR and suggest optimizations."""
    openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"🔍 Analyzing repository: {GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}")

    # Step 1: Get the most recent open pull request
    print("\n📋 Fetching most recent open pull request...")

    pr_number = None
    pr_found = False
    max_retries = 3

    for retry in range(max_retries):
        if retry > 0:
            print(
                f"   Retry {retry}/{max_retries-1} - Looking for open pull requests...")

        messages = [
            {"role": "user", "content": f"""Use the list_pull_requests tool to get all open pull requests for the repository {GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}.

IMPORTANT: You must call the list_pull_requests tool with:
- owner: {GITHUB_REPO_OWNER}
- repo: {GITHUB_REPO_NAME}
- state: open

After getting the list, tell me the number of the most recent open pull request."""}
        ]

        for _ in range(10):
            response = await openai.chat.completions.create(
                messages=messages,
                model="gpt-5-mini",
                tools=session["tools"]
            )

            choice = response.choices[0]
            tool_calls = choice.message.tool_calls

            if not tool_calls:
                # Try to extract PR number from response
                content = choice.message.content
                print(f"   Response: {content}")

                # Check if explicitly says no open PRs
                if "no open" in content.lower() or "no pull request" in content.lower():
                    print("   ℹ️  No open pull requests found")
                    pr_found = False
                    break

                # Look for PR number in response
                pr_match = re.search(r'#(\d+)', content)
                if pr_match:
                    pr_number = int(pr_match.group(1))
                    print(f"✅ Found PR #{pr_number}")
                    pr_found = True
                    break

                # Look for just a number
                number_match = re.search(r'\b(\d+)\b', content)
                if number_match:
                    pr_number = int(number_match.group(1))
                    print(f"✅ Found PR #{pr_number}")
                    pr_found = True
                    break

                break

            # Execute tools through Metorial
            tool_responses = await session["callTools"](tool_calls)

            # Add to conversation
            messages.append({
                "role": "assistant",
                "tool_calls": choice.message.tool_calls
            })
            messages.extend(tool_responses)

            # Check if we got PR info in the tool responses
            for resp in tool_responses:
                if resp.get('role') == 'tool':
                    content = resp.get('content', '')

                    # Check for empty list or no PRs
                    if content.strip() == '[]' or 'no pull requests' in content.lower():
                        print("   ℹ️  No open pull requests found in response")
                        pr_found = False
                        break

                    # Try to parse as JSON to extract PR info
                    try:
                        data = json.loads(content)
                        if isinstance(data, list) and len(data) > 0:
                            # Get the first (most recent) PR
                            first_pr = data[0]
                            if isinstance(first_pr, dict) and 'number' in first_pr:
                                pr_number = int(first_pr['number'])
                                print(f"✅ Found PR #{pr_number}")
                                pr_found = True
                                break
                    except json.JSONDecodeError:
                        pass

                    # Fallback: regex search for PR number
                    pr_match = re.search(r'"number":\s*(\d+)', content)
                    if pr_match:
                        pr_number = int(pr_match.group(1))
                        print(f"✅ Found PR #{pr_number}")
                        pr_found = True
                        break

            if pr_found:
                break

        if pr_found and pr_number:
            break

        # If we haven't found a PR and have retries left, wait and try again
        if retry < max_retries - 1:
            print("   ⏳ Waiting 2 seconds before retry...")
            await asyncio.sleep(2)

    if not pr_number or not pr_found:
        print("❌ Could not find an open pull request after retries")
        print("   Please ensure there is at least one open pull request in the repository")
        return

    # Step 2: Get all Python files changed in the PR
    print(f"\n📂 Fetching Python files changed in PR #{pr_number}...")

    messages = [
        {"role": "user", "content": f"""Use the appropriate tools to get all Python files changed in pull request #{pr_number} for repository {GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}.

CRITICAL INSTRUCTIONS:
1. First, use the tool to list all files changed in PR #{pr_number}
2. Get the PR's head branch/ref information (the branch with the changes)
3. For EACH Python (.py) file, use the tool to get its FULL RAW CONTENT from the PR's HEAD BRANCH (not the base branch)
4. YOU MUST include the complete file contents in your final response

RESPONSE FORMAT:
After fetching all files, respond with a JSON array like this:
[
  {{
    "filename": "path/to/file.py",
    "content": "the complete raw file content here - EVERY LINE of code"
  }},
  ...
]

CRITICAL REQUIREMENTS:
- Fetch contents from the PR's HEAD/SOURCE branch, NOT the base branch
- Include the ENTIRE file content, not summaries
- Only include Python (.py) files
- The "content" field must have the COMPLETE file contents verbatim

Do not summarize or truncate file contents. Include every line of code."""}
    ]

    py_files = {}
    files_found = False

    for iteration in range(20):
        response = await openai.chat.completions.create(
            messages=messages,
            model="gpt-5-mini",
            tools=session["tools"]
        )

        choice = response.choices[0]
        tool_calls = choice.message.tool_calls

        if not tool_calls:
            content = choice.message.content
            print(f"   Response: {content[:200]}...")  # Show truncated preview

            # Try to parse as JSON array (our requested format)
            try:
                # Remove markdown code blocks if present
                json_content = content
                if "```json" in json_content or "```" in json_content:
                    # Extract content between code blocks
                    json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', json_content, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1)

                # Try to parse as JSON
                files_data = json.loads(json_content)

                if isinstance(files_data, list):
                    for file_info in files_data:
                        if isinstance(file_info, dict):
                            filename = file_info.get('filename', '')
                            file_content = file_info.get('content', '')
                            if filename and filename.endswith('.py') and file_content:
                                py_files[filename] = file_content
                                print(f"   ✅ Extracted {filename} ({len(file_content)} bytes)")
                                files_found = True

                    if files_found:
                        print(f"   ✅ Successfully parsed {len(py_files)} file(s) from JSON response")
                        break
            except (json.JSONDecodeError, ValueError) as e:
                # Not valid JSON, try fallback patterns
                print(f"   ⚠️  JSON parse failed: {e}, trying fallback extraction...")

                # Fallback: Look for pattern like "Filename: X\nContent:\n[code]"
                filename_pattern = r'Filename:\s+(\S+\.py)\s+Content:\s*\n(.*?)(?=\n\nFilename:|$)'
                matches = re.finditer(filename_pattern, content, re.DOTALL)

                for match in matches:
                    filename = match.group(1)
                    file_content = match.group(2).strip()
                    if file_content:
                        py_files[filename] = file_content
                        print(f"   ✅ Extracted content for {filename} from text pattern")
                        files_found = True

            # Check if it says no Python files
            if "no python" in content.lower() or "no .py" in content.lower():
                if not py_files:
                    print("   ℹ️  No Python files found in PR")
                    files_found = False
            elif py_files:
                files_found = True
            break

        # Execute tools through Metorial
        tool_responses = await session["callTools"](tool_calls)

        # Add to conversation
        messages.append({
            "role": "assistant",
            "tool_calls": choice.message.tool_calls
        })
        messages.extend(tool_responses)

        # Try to extract Python file contents from tool responses
        for resp in tool_responses:
            if resp.get('role') == 'tool':
                content = resp.get('content', '')

                # Try to parse as JSON to extract file info
                try:
                    data = json.loads(content)

                    # Handle list of files
                    if isinstance(data, list):
                        for file_info in data:
                            if isinstance(file_info, dict):
                                filename = file_info.get(
                                    'filename', file_info.get('path', ''))
                                if filename.endswith('.py'):
                                    # Check if we have content in various possible fields
                                    file_content = file_info.get('content') or \
                                                   file_info.get('raw_content') or \
                                                   file_info.get('data') or \
                                                   file_info.get('text') or ''
                                    if file_content:
                                        py_files[filename] = file_content
                                        print(f"   ✅ Got content for {filename} ({len(file_content)} bytes)")
                                        files_found = True
                                    else:
                                        print(f"   ⚠️  Found {filename} but no content field")

                    # Handle single file
                    elif isinstance(data, dict):
                        filename = data.get('filename', data.get('path', ''))
                        if filename and filename.endswith('.py'):
                            # Check multiple possible content fields
                            file_content = data.get('content') or \
                                           data.get('raw_content') or \
                                           data.get('data') or \
                                           data.get('text') or ''
                            if file_content:
                                py_files[filename] = file_content
                                print(f"   ✅ Got content for {filename} ({len(file_content)} bytes)")
                                files_found = True
                            else:
                                print(f"   ⚠️  Found {filename} but no content field")

                except json.JSONDecodeError:
                    # Not JSON, might be raw file content
                    # Check if this looks like Python code
                    if any(keyword in content for keyword in ['def ', 'class ', 'import ', 'from ']):
                        # Try to infer filename from previous tool calls
                        # This is a fallback for raw content responses
                        if tool_calls:
                            for tool_call in choice.message.tool_calls:
                                if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'arguments'):
                                    try:
                                        args = json.loads(
                                            tool_call.function.arguments)
                                        path = args.get('path', args.get(
                                            'file', args.get('filename', '')))
                                        if path and path.endswith('.py'):
                                            py_files[path] = content
                                            print(
                                                f"   ✅ Got raw content for {path}")
                                            files_found = True
                                    except Exception:
                                        pass

        # If we have Python files and finished recent tool calls, we can break
        if py_files and iteration > 5:
            print(f"   Found {len(py_files)} Python file(s) so far")

    print(f"\n✅ Found {len(py_files)} Python file(s) total")

    if not py_files or not files_found:
        print("⚠️  No Python files found in PR. Exiting.")
        return

    # Step 3: Extract and optimize code from each file
    print("\n🔬 Analyzing code for optimization opportunities...")
    all_optimizations = []

    for file_path, content in py_files.items():
        print(f"\n  Analyzing {file_path}...")

        # Extract optimizable code segments
        segments = extract_optimizable_code(file_path, content)

        print(f"    Found {len(segments)} code segment(s) to analyze")

        # Process each segment
        for segment in segments:
            try:
                if segment['type'] == 'python':
                    # For Python code, identify individual speedup opportunities
                    print(f"    Analyzing {segment['description']}...")
                    print(f"      Function starts at line {segment.get('line_start', 1)} in file")

                    speedups = identify_individual_speedups(
                        segment['code'],
                        file_path,
                        segment.get('line_start', 1)
                    )

                    print(f"      Found {len(speedups)} individual speedup(s)")

                    # Optimize each individual speedup
                    for speedup in speedups:
                        try:
                            issue_preview = speedup['issue'][:50]
                            print(
                                f"        Optimizing: {issue_preview}...")

                            # Run optimization on the specific code snippet
                            result = optimize(
                                speedup['code_snippet'], benchmark=False)

                            # DEBUG: Print the result to see what optimize() returned
                            print(f"\n          DEBUG - Optimize result:")
                            print(f"          Type: {result.get('type')}")
                            print(f"          Original: {result.get('original')[:100]}..." if len(result.get('original', '')) > 100 else f"          Original: {result.get('original')}")
                            print(f"          Optimized: {result.get('optimized')[:100]}..." if len(result.get('optimized', '')) > 100 else f"          Optimized: {result.get('optimized')}")
                            print(f"          Improvements: {result.get('improvements')}")
                            print(f"          Speedup: {result.get('speedup')}")
                            print(f"          Validation: {result.get('validation')}\n")

                            # Add the speedup as a separate suggestion
                            if result['optimized'] != result['original']:
                                all_optimizations.append({
                                    'location': speedup['location'],
                                    'description': speedup['issue'],
                                    'suggestion': speedup['suggestion'],
                                    'result': result,
                                    'type': 'speedup'
                                })
                                print("          ✅ Optimization created!")
                            else:
                                print("          ℹ️  No change needed")

                        except Exception as e:
                            print(f"          ⚠️  Error: {e}")

                elif segment['type'] == 'sql':
                    # For SQL, optimize the entire query
                    print(f"    Optimizing {segment['description']}...")
                    result = optimize(segment['code'], benchmark=False)

                    # DEBUG: Print the result
                    print(f"\n      DEBUG - SQL Optimize result:")
                    print(f"      Type: {result.get('type')}")
                    print(f"      Improvements: {result.get('improvements')}")
                    print(f"      Validation: {result.get('validation')}\n")

                    if result['optimized'] != result['original']:
                        all_optimizations.append({
                            'location': segment['location'],
                            'description': segment['description'],
                            'result': result,
                            'type': 'sql',
                            'full_lines': segment.get('full_lines', ''),
                            'original_sql': segment.get('original_sql', result['original']),
                            'optimized_sql': result['optimized']
                        })
                        print("      ✅ Optimization found!")
                    else:
                        print("      ℹ️  No optimization needed")

                elif segment['type'] == 'regex':
                    # For regex, optimize the pattern WITH benchmarking to get performance data
                    print(f"    Optimizing {segment['description']}...")
                    result = optimize(segment['code'], benchmark=True)

                    # DEBUG: Print the result
                    print(f"\n      DEBUG - Regex Optimize result:")
                    print(f"      Type: {result.get('type')}")
                    print(f"      Original: {result.get('original')}")
                    print(f"      Optimized: {result.get('optimized')}")
                    print(f"      Speedup: {result.get('speedup')}")
                    print(f"      Improvements: {result.get('improvements')}")
                    print(f"      Validation: {result.get('validation')}\n")

                    if result['optimized'] != result['original']:
                        all_optimizations.append({
                            'location': segment['location'],
                            'description': segment['description'],
                            'result': result,
                            'type': 'regex',
                            'full_line': segment.get('full_line', ''),
                            'original_pattern': result['original'],
                            'optimized_pattern': result['optimized']
                        })
                        print("      ✅ Optimization found!")
                    else:
                        print("      ℹ️  No optimization needed")

            except Exception as e:
                print(f"      ⚠️  Error processing segment: {e}")

    print(f"\n✅ Found {len(all_optimizations)} optimization suggestion(s)")

    # Step 4: Post suggestions as individual code suggestion review comments
    if all_optimizations:
        print("\n💬 Posting code suggestion review comments to PR...")

        # Format each optimization as a code suggestion
        suggestions = []
        for opt in all_optimizations:
            suggestion = format_code_suggestion(opt)
            if suggestion:
                suggestions.append(suggestion)

        print(f"   Formatted {len(suggestions)} code suggestion(s)")

        # Post each suggestion as a review comment
        comments_posted = 0
        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n   Posting suggestion {i}/{len(suggestions)} for {suggestion['path']}:{suggestion['line']}...")

            messages = [
                {"role": "user", "content": f"""Use the create_review_comment or add_review_comment tool to post a review comment on pull request #{pr_number} in repository {GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}.

IMPORTANT: Call the appropriate tool with these EXACT parameters:
- owner: {GITHUB_REPO_OWNER}
- repo: {GITHUB_REPO_NAME}
- pull_request_number or pull_number: {pr_number}
- path: {suggestion['path']}
- line: {suggestion['line']}
- body: (the exact comment text below - DO NOT modify or add any additional content)

Comment body to post:
{suggestion['body']}

If the create_review_comment tool is not available, try these alternatives in order:
1. create_pull_request_review_comment
2. add_comment_to_pull_request_file
3. Any other tool that can add a line-specific comment to a PR

CRITICAL: The comment MUST be posted on line {suggestion['line']} of file {suggestion['path']}, not as a general PR comment."""}
            ]

            comment_posted = False

            for attempt in range(10):
                response = await openai.chat.completions.create(
                    messages=messages,
                    model="gpt-5-mini",
                    tools=session["tools"]
                )

                choice = response.choices[0]
                tool_calls = choice.message.tool_calls

                if not tool_calls:
                    content = choice.message.content
                    print(f"      Response: {content[:100]}...")

                    # Check if comment was posted
                    if "comment" in content.lower() and ("posted" in content.lower() or "added" in content.lower() or "created" in content.lower()):
                        print(f"      ✅ Suggestion {i} posted successfully!")
                        comment_posted = True
                        comments_posted += 1
                    break

                tool_responses = await session["callTools"](tool_calls)

                messages.append({
                    "role": "assistant",
                    "tool_calls": choice.message.tool_calls
                })
                messages.extend(tool_responses)

                # Check if comment was posted in responses
                for resp in tool_responses:
                    if resp.get('role') == 'tool':
                        resp_content = resp.get('content', '')
                        if resp_content and ('id' in resp_content or 'created' in resp_content.lower()):
                            print(f"      ✅ Suggestion {i} posted successfully!")
                            comment_posted = True
                            comments_posted += 1
                            break

                if comment_posted:
                    break

            if not comment_posted:
                print(f"      ⚠️  Could not confirm suggestion {i} was posted")

        print(f"\n✅ Posted {comments_posted}/{len(suggestions)} code suggestion(s) to PR")
    else:
        print("\n✨ No optimization opportunities found - code looks great!")


async def main():
    """Main execution function for PR optimization"""

    # Initialize Metorial SDK
    metorial = Metorial(api_key=os.getenv("METORIAL_API_KEY"))

    # Get deployment ID from environment
    github_deployment_id = os.getenv("GITHUB_METORIAL_DEPLOYMENT_ID")

    if not github_deployment_id:
        print("❌ Error: GITHUB_METORIAL_DEPLOYMENT_ID not set in environment")
        return

    oauth_session = await setup_oauth_if_needed(metorial, github_deployment_id)

    try:
        await metorial.with_provider_session(
            MetorialOpenAI.chat_completions,
            [
                {
                    "serverDeploymentId": github_deployment_id,
                    "oauthSessionId": oauth_session.id
                }
            ],
            session_action
        )

    except MetorialAPIError as e:
        print(f"❌ Metorial API Error: {e.message} (Status: {e.status_code})")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
