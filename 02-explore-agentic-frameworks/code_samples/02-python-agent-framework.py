import asyncio
import os
import sys
from typing import Annotated

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

chat_client = OpenAIChatClient(
    model=deployment,
    azure_endpoint=endpoint,
    api_key=api_key,
)


@tool(approval_mode="never_require")
def check_destination_availability(
    destination: Annotated[str, "The destination to check availability for"],
) -> str:
    """Check if a vacation destination is currently available for booking."""
    available = {
        "Barcelona": True,
        "Tokyo": True,
        "Cape Town": False,
        "Vancouver": True,
        "Dubai": False,
    }
    is_available = available.get(destination, False)
    return f"{destination} is {'available' if is_available else 'not available'} for booking."


agent = chat_client.as_agent(
    name="TravelAvailabilityAgent",
    instructions=(
        "You are a travel booking agent. Help users check destination availability "
        "and make recommendations. Always check availability before recommending a destination."
    ),
    tools=[check_destination_availability],
)


async def main() -> None:
    session = agent.create_session()

    response = await agent.run(
        "Which destinations do you have available?",
        session=session,
    )
    print(f"Agent: {response}")

    print()

    response = await agent.run(
        "I'd like to go somewhere warm. What's available?",
        session=session,
    )
    print(f"Agent: {response}")


if __name__ == "__main__":
    asyncio.run(main())
