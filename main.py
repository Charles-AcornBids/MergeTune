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

from openai import AsyncOpenAI
from metorial import Metorial, MetorialOpenAI, MetorialAPIError

from optimize import optimize, _extract_sql_from_code, _is_regex_pattern
from openai import OpenAI

# Load environment variables
load_dotenv()

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
2. The line numbers where it appears (relative to the start of the code)
3. A brief description of the performance issue
4. A suggested optimization approach

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
        for speedup in speedups:
            speedup['line_start'] = start_line + speedup['line_start'] - 1
            speedup['line_end'] = start_line + speedup['line_end'] - 1
            speedup['location'] = f"{file_path}:{speedup['line_start']}-{speedup['line_end']}"

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
                start_line = node.lineno - 1  # 0-indexed
                end_line = node.end_lineno if hasattr(
                    node, 'end_lineno') else start_line + 1

                func_code = '\n'.join(lines[start_line:end_line])

                functions.append({
                    'name': node.name,
                    'code': func_code,
                    'line_start': node.lineno,
                    'line_end': end_line + 1
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

    # Extract SQL queries
    sql_queries = _extract_sql_from_code(content)
    for sql, start_line, end_line in sql_queries:
        optimizable.append({
            'type': 'sql',
            'code': sql,
            'location': f"{file_path}:{start_line}-{end_line}",
            'description': f"SQL query at lines {start_line}-{end_line}"
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
    for match in re.finditer(regex_pattern, content):
        potential_regex = match.group(1)
        if _is_regex_pattern(potential_regex):
            line_num = content[:match.start()].count('\n') + 1
            optimizable.append({
                'type': 'regex',
                'code': potential_regex,
                'location': f"{file_path}:{line_num}",
                'description': f"Regex pattern at line {line_num}"
            })

    return optimizable


def format_optimization_suggestions(optimizations: List[Dict[str, Any]]) -> str:
    """
    Format optimization suggestions as a markdown comment for GitHub PR.

    Args:
        optimizations: List of optimization results

    Returns:
        Formatted markdown string
    """
    if not optimizations:
        return "No significant optimization opportunities found. Great work! 🎉"

    comment = "# 🚀 MergeTune Optimization Suggestions\n\n"
    comment += f"Found **{len(optimizations)}** optimization opportunities in this PR:\n\n"

    for i, opt in enumerate(optimizations, 1):
        result = opt['result']
        location = opt['location']
        description = opt['description']
        opt_type = opt.get('type', result.get('type', 'unknown'))

        comment += f"## {i}. {description}\n\n"
        comment += f"**Location:** `{location}`\n\n"

        # Add type-specific details
        if opt_type == 'speedup':
            # Individual Python speedup opportunity
            comment += "**Type:** Performance Speedup\n\n"

            if opt.get('suggestion'):
                comment += f"**Suggestion:** {opt['suggestion']}\n\n"

            comment += "**Current Code:**\n```python\n" + \
                result['original'].strip() + "\n```\n\n"
            comment += "**Optimized Code:**\n```python\n" + \
                result['optimized'].strip() + "\n```\n\n"

            if result.get('improvements'):
                comment += "**Improvements:**\n"
                for imp in result['improvements']:
                    comment += f"- {imp}\n"
                comment += "\n"

        elif opt_type == 'sql' or result['type'] == 'sql':
            comment += "**Type:** SQL Query Optimization\n\n"

            if result.get('validation'):
                val = result['validation']
                orig_score = val['original_complexity']['score']
                opt_score = val['optimized_complexity']['score']
                improvement = val['estimated_improvement']

                comment += f"**Complexity Score:** {orig_score} → {opt_score} ({improvement:.1f}% improvement)\n\n"

            comment += "**Original Query:**\n```sql\n" + \
                result['original'].strip() + "\n```\n\n"
            comment += "**Optimized Query:**\n```sql\n" + \
                result['optimized'].strip() + "\n```\n\n"

            if result.get('improvements'):
                comment += "**Improvements:**\n"
                for imp in result['improvements']:
                    comment += f"- {imp}\n"
                comment += "\n"

        elif opt_type == 'regex' or result['type'] == 'regex':
            comment += "**Type:** Regex Pattern Optimization\n\n"
            comment += f"**Original Pattern:** `{result['original']}`\n\n"
            comment += f"**Optimized Pattern:** `{result['optimized']}`\n\n"

            if result.get('speedup') and result['speedup'] > 1:
                comment += f"**Performance:** {result['speedup']:.2f}x faster\n\n"

        elif result['type'] == 'python':
            # Fallback for general Python optimization
            comment += "**Type:** Python Code Optimization\n\n"
            comment += "**Original Code:**\n```python\n" + \
                result['original'].strip() + "\n```\n\n"
            comment += "**Optimized Code:**\n```python\n" + \
                result['optimized'].strip() + "\n```\n\n"

            if result.get('improvements'):
                comment += "**Improvements:**\n"
                for imp in result['improvements']:
                    comment += f"- {imp}\n"
                comment += "\n"

        comment += "---\n\n"

    comment += "\n*Generated by MergeTune - AI-powered code optimization*"

    return comment


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
        {"role": "user", "content": f"""Use the appropriate tool to get all files changed in pull request #{pr_number} for repository {GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}.

IMPORTANT: Call the tool to list files in the pull request, then for each Python (.py) file, get its full content FROM THE PR'S HEAD BRANCH (the branch with the changes), NOT from the default branch.

Steps:
1. List all files changed in PR #{pr_number}
2. Get the PR's head branch/ref information
3. For each file that ends with .py, get its full raw content FROM THE PR'S HEAD BRANCH
4. Return the filenames and contents

CRITICAL: You MUST fetch file contents from the pull request's source/head branch, not the base/default branch. Use the PR's head ref when fetching file contents.

Focus only on Python (.py) files."""}
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
            print(f"   Response: {content}")

            # Try to extract file content from the agent's summary message
            # Look for pattern like "Filename: X\nContent:\n[code]"
            filename_pattern = r'Filename:\s+(\S+\.py)\s+Content:\s*\n(.*?)(?=\n\nFilename:|$)'
            matches = re.finditer(filename_pattern, content, re.DOTALL)

            for match in matches:
                filename = match.group(1)
                file_content = match.group(2).strip()
                if file_content:
                    py_files[filename] = file_content
                    print(f"   ✅ Extracted content for {filename} from summary")
                    files_found = True

            # Check if it says no Python files
            if "no python" in content.lower() or "no .py" in content.lower():
                # Only set files_found to False if we haven't found any files yet
                if not py_files:
                    print("   ℹ️  No Python files found in PR")
                    files_found = False
            elif py_files:
                # If we already have files, we're done
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

        # Try to extract Python file contents
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
                                    # Check if we have content
                                    file_content = file_info.get(
                                        'content', file_info.get('raw_content', ''))
                                    if file_content:
                                        py_files[filename] = file_content
                                        print(
                                            f"   ✅ Got content for {filename}")
                                        files_found = True

                    # Handle single file
                    elif isinstance(data, dict):
                        filename = data.get('filename', data.get('path', ''))
                        if filename and filename.endswith('.py'):
                            file_content = data.get(
                                'content', data.get('raw_content', ''))
                            if file_content:
                                py_files[filename] = file_content
                                print(f"   ✅ Got content for {filename}")
                                files_found = True

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

                    speedups = identify_individual_speedups(
                        segment['code'],
                        file_path,
                        segment.get('line_start', 1)
                    )

                    print(f"      Found {len(speedups)} individual speedup(s)")

                    # Optimize each individual speedup
                    for speedup in speedups:
                        try:
                            print(
                                f"        Optimizing: {speedup['issue'][:50]}...")

                            # Run optimization on the specific code snippet
                            result = optimize(
                                speedup['code_snippet'], benchmark=False)

                            # Add the speedup as a separate suggestion
                            if result['optimized'] != result['original']:
                                all_optimizations.append({
                                    'location': speedup['location'],
                                    'description': f"{speedup['issue']}",
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

                    if result['optimized'] != result['original']:
                        all_optimizations.append({
                            'location': segment['location'],
                            'description': segment['description'],
                            'result': result,
                            'type': 'sql'
                        })
                        print("      ✅ Optimization found!")
                    else:
                        print("      ℹ️  No optimization needed")

                elif segment['type'] == 'regex':
                    # For regex, optimize the pattern
                    print(f"    Optimizing {segment['description']}...")
                    result = optimize(segment['code'], benchmark=False)

                    if result['optimized'] != result['original']:
                        all_optimizations.append({
                            'location': segment['location'],
                            'description': segment['description'],
                            'result': result,
                            'type': 'regex'
                        })
                        print("      ✅ Optimization found!")
                    else:
                        print("      ℹ️  No optimization needed")

            except Exception as e:
                print(f"      ⚠️  Error processing segment: {e}")

    print(f"\n✅ Found {len(all_optimizations)} optimization suggestion(s)")

    # Step 4: Format and post suggestions as a comment
    if all_optimizations:
        print("\n💬 Posting optimization suggestions to PR...")

        comment_body = format_optimization_suggestions(all_optimizations)

        messages = [
            {"role": "user", "content": f"""Use the create_issue_comment or add_comment_to_pull_request tool to post a comment on pull request #{pr_number} in repository {GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}.

IMPORTANT: Call the appropriate tool with:
- owner: {GITHUB_REPO_OWNER}
- repo: {GITHUB_REPO_NAME}
- pull_request_number or issue_number: {pr_number}
- body: (the comment text below)

Comment body to post:
{comment_body}"""}
        ]

        comment_posted = False

        for _ in range(10):
            response = await openai.chat.completions.create(
                messages=messages,
                model="gpt-5-mini",
                tools=session["tools"]
            )

            choice = response.choices[0]
            tool_calls = choice.message.tool_calls

            if not tool_calls:
                content = choice.message.content
                print(f"   Response: {content}")

                # Check if comment was posted
                if "comment" in content.lower() and ("posted" in content.lower() or "added" in content.lower() or "created" in content.lower()):
                    print("✅ Comment posted successfully!")
                    comment_posted = True
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
                    content = resp.get('content', '')
                    if content and ('id' in content or 'created' in content.lower()):
                        print("✅ Comment posted successfully!")
                        comment_posted = True
                        break

            if comment_posted:
                break

        if not comment_posted:
            print("⚠️  Could not confirm comment was posted")
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
