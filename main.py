"""
MergeTune - AI-powered GitHub repository management using Metorial SDK
"""
import asyncio
import os
from dotenv import load_dotenv

from openai import AsyncOpenAI
from metorial import Metorial, MetorialOpenAI, MetorialAPIError

# Load environment variables
load_dotenv()


async def setup_oauth_if_needed(metorial: Metorial, deployment_id: str) -> None:
    """
    Set up OAuth authentication for deployments that require it (e.g., GitHub).
    Uncomment this function call in main() if OAuth is needed.
    """
    print("🔗 Creating OAuth session...")
    oauth_session = metorial.oauth.sessions.create(
        server_deployment_id=deployment_id
    )

    print(f"📋 Please authenticate at: {oauth_session.url}")

    print("\n⏳ Waiting for OAuth completion...")
    await metorial.oauth.wait_for_completion([oauth_session])

    print("✅ OAuth session completed!\n")


async def main():
    """Main execution function for listing GitHub repos"""

    # Initialize Metorial SDK
    metorial = Metorial(api_key=os.getenv("METORIAL_API_KEY"))

    # Initialize OpenAI client
    openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Get deployment ID from environment
    github_deployment_id = os.getenv("GITHUB_METORIAL_DEPLOYMENT_ID")

    if not github_deployment_id:
        print("❌ Error: GITHUB_METORIAL_DEPLOYMENT_ID not set in environment")
        return

    # Uncomment if OAuth authentication is required:
    # await setup_oauth_if_needed(metorial, github_deployment_id)

    try:
        response = await metorial.run(
            message="List all my GitHub repositories.",
            server_deployments=[github_deployment_id],
            client=openai,
            model="gpt-4o",
            max_steps=25  # optional
        )

        print("Response:", response.text)

    except MetorialAPIError as e:
        print(f"❌ Metorial API Error: {e.message} (Status: {e.status_code})")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
