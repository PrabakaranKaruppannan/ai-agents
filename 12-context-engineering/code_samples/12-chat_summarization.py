import asyncio
import os
import sys

import dotenv
from agent_framework import tool
from agent_framework.openai import OpenAIChatClient

# Avoid Windows console UnicodeEncodeError when printing responses.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

dotenv.load_dotenv(dotenv.find_dotenv())

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

if not endpoint or not deployment or not api_key:
    raise ValueError(
        "Missing required environment variables. "
        "Please set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, "
        "and AZURE_OPENAI_API_KEY in your .env file."
    )

chat_client = OpenAIChatClient(
    model=deployment,
    azure_endpoint=endpoint,
    api_key=api_key,
)


@tool(approval_mode="never_require")
def summarize_preferences(conversation_notes: str) -> str:
    """Summarize accumulated user preferences into a compact format."""
    return f"[SUMMARY] User preferences recorded: {conversation_notes}"


async def run_context_aware_conversation() -> None:
    """Multi-turn session showing context retention across preference changes."""
    print("=== Context-Aware Travel Agent ===\n")

    agent = chat_client.as_agent(
        name="ContextAwareAgent",
        instructions="""You are a helpful travel planning assistant with excellent memory management.
When conversations get long:
1. Summarize previous context into key points
2. Track user preferences mentioned earlier
3. Reference previous decisions without repeating full details
Always maintain continuity while being concise.""",
    )

    session = agent.create_session()

    turns = [
        "I'm planning a trip to Japan. I love sushi, temples, and photography.",
        "My budget is $3000 and I'll be traveling solo for 10 days in April.",
        "Based on everything I've told you so far, what's the one thing you'd recommend I not miss?",
        "What about accommodation? I prefer traditional Japanese inns.",
        "Actually, I've changed my mind about the dates. I'll go in October instead for the autumn colors.",
        "Summarize my complete travel plan so far — destination, budget, duration, interests, accommodation, and timing.",
    ]

    for i, user_message in enumerate(turns, start=1):
        print(f"User (Turn {i}): {user_message}\n")
        response = await agent.run(user_message, session=session)
        print(f"Turn {i}: {response}\n")
        print("-" * 60 + "\n")


async def run_summarization_pattern() -> None:
    """Agent uses summarize_preferences to record compact context."""
    print("=== Context Summarization Pattern ===\n")

    summarizing_agent = chat_client.as_agent(
        name="SummarizingTravelAgent",
        instructions="""You are a helpful travel planning assistant that actively manages conversation context.

CONTEXT MANAGEMENT RULES:
1. After gathering several user preferences, call summarize_preferences() to record a compact summary
2. When the user asks you to recall details, reference your recorded summaries
3. Keep responses concise — avoid restating the entire history

PLANNING PROCESS:
1. Gather user preferences (destination, budget, dates, interests)
2. Summarize preferences using the tool
3. Create recommendations based on the summary
4. Update the summary when preferences change""",
        tools=[summarize_preferences],
    )

    summary_session = summarizing_agent.create_session()

    response = await summarizing_agent.run(
        "I want to visit Greece. I love seafood, history, and island hopping. "
        "Budget is $4000 for two weeks. Traveling with my partner in June. "
        "Please record these preferences using your summarization tool.",
        session=summary_session,
    )
    print(f"Agent: {response}\n")

    response = await summarizing_agent.run(
        "Now, based on what you've recorded, suggest the top 3 islands we should visit.",
        session=summary_session,
    )
    print(f"Agent: {response}\n")


async def main() -> None:
    await run_context_aware_conversation()
    print("=" * 60 + "\n")
    await run_summarization_pattern()


if __name__ == "__main__":
    asyncio.run(main())
