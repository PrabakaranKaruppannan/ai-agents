import asyncio
import os
import sys
import time
from typing import Annotated

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
def get_flight_info(destination: Annotated[str, "The destination city"]) -> str:
    """Get flight information for a destination."""
    flights = {
        "Paris": "BA 304, 08:30-11:45, $350",
        "Tokyo": "JL 044, 11:00-07:00+1, $890",
        "Barcelona": "VY 7821, 07:15-10:30, $280",
    }
    return flights.get(destination, f"No flights found to {destination}")


@tool(approval_mode="never_require")
def get_activity_suggestions(
    destination: Annotated[str, "The destination city"],
) -> str:
    """Get activity suggestions for a destination."""
    activities = {
        "Paris": (
            "Louvre Museum, Eiffel Tower, Seine River Cruise, Montmartre walking tour"
        ),
        "Tokyo": (
            "Senso-ji Temple, Tsukiji Market tour, Shibuya Crossing, teamLab Borderless"
        ),
        "Barcelona": (
            "Sagrada Familia, Park Güell, La Boqueria Market, Gothic Quarter walk"
        ),
    }
    return activities.get(destination, f"No activities found for {destination}")


async def run_observable_travel_agent() -> str:
    """Run the travel agent with simple latency observability."""
    print("=== Observable Travel Agent ===\n")

    agent = chat_client.as_agent(
        tools=[get_flight_info, get_activity_suggestions],
        name="TravelAgent",
        instructions=(
            "You are a helpful travel agent. Use the available tools to help users "
            "plan their trips. Provide comprehensive, actionable travel advice."
        ),
    )

    start_time = time.time()
    response = await agent.run(
        "I want to plan a day trip in Paris. What flights and activities do you recommend?"
    )
    elapsed = time.time() - start_time
    print(f"Response ({elapsed:.2f}s):\n{response}")
    return str(response)


async def run_evaluation(prior_response: str) -> None:
    """Evaluate the travel agent response with a second agent."""
    print("\n=== Response Evaluation ===\n")

    evaluator = chat_client.as_agent(
        name="ResponseEvaluator",
        instructions="""You evaluate travel agent responses on these criteria:
1. Completeness (1-5): Did it cover flights AND activities?
2. Accuracy (1-5): Is the information consistent?
3. Helpfulness (1-5): Would a traveler find this actionable?
4. Overall Score (1-5)
Provide scores and a brief explanation for each.""",
    )

    evaluation = await evaluator.run(
        f"Evaluate this travel agent response:\n\n{prior_response}"
    )
    print(f"Evaluation:\n{evaluation}")
    print()


def print_cost_strategies() -> None:
    print("=== Cost Management Strategies ===\n")
    print(
        "Prompt optimization | Keep system instructions concise; remove redundant context.\n"
        "Model selection     | Use smaller models for simple tasks; larger for complex reasoning.\n"
        "Caching             | Cache tool results and frequent queries.\n"
        "Token budgets       | Set max_tokens limits to avoid unexpectedly long responses.\n"
        "Batching            | Group multiple queries into a single API call where possible.\n"
    )


async def main() -> None:
    prior_response = await run_observable_travel_agent()
    print("\n" + "-" * 60)
    await run_evaluation(prior_response)
    print("-" * 60 + "\n")
    print_cost_strategies()


if __name__ == "__main__":
    asyncio.run(main())
