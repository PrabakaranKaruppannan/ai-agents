import asyncio
import os
import random
import sys

import dotenv
from agent_framework import tool
from agent_framework.openai import OpenAIChatClient

# Avoid Windows console UnicodeEncodeError when streaming responses.
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

# A list of vacation destinations the tool can choose from.
_DESTINATIONS = [
    "Barcelona, Spain",
    "Paris, France",
    "Berlin, Germany",
    "Tokyo, Japan",
    "Sydney, Australia",
    "New York, USA",
    "Cairo, Egypt",
    "Cape Town, South Africa",
    "Rio de Janeiro, Brazil",
    "Bali, Indonesia",
]

# Track the last destination so repeated calls avoid immediate repeats.
_last_destination: str | None = None


@tool(approval_mode="never_require")
def get_random_destination() -> str:
    """Provides a random vacation destination."""
    global _last_destination
    available = _DESTINATIONS.copy()
    if _last_destination and len(available) > 1:
        available.remove(_last_destination)
    destination = random.choice(available)
    _last_destination = destination
    return destination


# OpenAIChatClient targets Azure OpenAI's v1 endpoint and uses the Responses API.
chat_client = OpenAIChatClient(
    model=deployment,
    azure_endpoint=endpoint,
    api_key=api_key,
)

agent = chat_client.as_agent(
    name="TravelAgent",
    instructions=(
        "You are a helpful AI Agent that can help plan vacations for customers "
        "at random destinations"
    ),
    tools=[get_random_destination],
)

user_inputs = [
    "Plan me a day trip.",
    "I don't like that destination. Plan me another vacation.",
]


async def main() -> None:
    # A session keeps conversation history across turns.
    session = agent.create_session()

    for user_input in user_inputs:
        print(f"User:\n  {user_input}\n")
        print("TravelAgent:")
        async for chunk in agent.run(user_input, session=session, stream=True):
            print(chunk, end="", flush=True)
        print("\n")
        print("-" * 60)
        print()


if __name__ == "__main__":
    asyncio.run(main())
