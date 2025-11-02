"""
MergeTune - AI-powered GitHub repository management using Metorial SDK
"""
import asyncio
import os
import json
from pathlib import Path
from dotenv import load_dotenv

from openai import AsyncOpenAI
from metorial import Metorial, MetorialOpenAI, MetorialAPIError

# Load environment variables
load_dotenv()

# OAuth cache file path
OAUTH_CACHE_FILE = Path(".oauth_cache.json")


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


async def session_action(session):
    # Initialize OpenAI client
    openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {"role": "user", "content": "View the most recent open pull request for Charles-AcornBids YC_Agent_Jam_Example repository and add a comment saying 'hey there from MergeTune'."}
    ]

    for i in range(10):
        response = await openai.chat.completions.create(
            messages=messages,
            model="gpt-4o",
            tools=session["tools"]
        )

        choice = response.choices[0]
        tool_calls = choice.message.tool_calls

        if not tool_calls:
            print(choice.message.content)
            return

        # Execute tools through Metorial
        tool_responses = await session["callTools"](tool_calls)

        # Add to conversation
        messages.append({
            "role": "assistant",
            "tool_calls": choice.message.tool_calls
        })
        messages.extend(tool_responses)


async def main():
    """Main execution function for listing GitHub repos"""

    # Initialize Metorial SDK
    metorial = Metorial(api_key=os.getenv("METORIAL_API_KEY"))

    # Get deployment ID from environment
    github_deployment_id = os.getenv("GITHUB_METORIAL_DEPLOYMENT_ID")

    if not github_deployment_id:
        print("❌ Error: GITHUB_METORIAL_DEPLOYMENT_ID not set in environment")
        return

    # Uncomment if OAuth authentication is required:
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
