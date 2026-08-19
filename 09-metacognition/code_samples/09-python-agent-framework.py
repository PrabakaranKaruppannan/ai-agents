import asyncio
import os
import sys
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
def get_flight_times(
    destination: Annotated[str, "The destination city"],
) -> str:
    """Get available flight times for a destination (primary source)."""
    flights = {
        "Paris": "Departures: 08:00, 12:30, 17:45 — from $350",
        "Tokyo": "Departures: 11:00, 23:30 — from $890",
        "Barcelona": "Departures: 07:15, 14:00, 19:30 — from $280",
    }
    if destination in flights:
        return flights[destination]
    raise Exception(f"404: No flights found for {destination} in primary system")


@tool(approval_mode="never_require")
def get_flight_times_backup(
    destination: Annotated[str, "The destination city"],
) -> str:
    """Get available flight times from backup system (used when primary fails)."""
    backup_flights = {
        "Berlin": "Departures: 09:00, 16:00 — from $220",
        "Sydney": "Departures: 22:00 — from $1200",
        "New York City": "Departures: 06:00, 10:30, 15:00, 20:00 — from $450",
    }
    return backup_flights.get(
        destination,
        f"No flights found for {destination} in any system. Please try again later.",
    )


async def run_flight_booking_tests() -> str:
    """Self-reflecting agent with primary/backup error recovery."""
    agent = chat_client.as_agent(
        tools=[get_flight_times, get_flight_times_backup],
        name="FlightBookingAgent",
        instructions="""You are a flight booking agent with self-reflection capabilities.

When looking up flights:
1. Try the primary flight system first (get_flight_times)
2. If the primary system fails (404 error), acknowledge the error and try the backup system (get_flight_times_backup)
3. Always explain to the user what happened — be transparent about fallbacks
4. If both systems fail, apologize and suggest alternatives

After each response, briefly evaluate whether your answer was complete and helpful.""",
    )

    print("=== Test 1: Destination in primary system ===")
    response = await agent.run("What flights are available to Paris?")
    print(response)

    print("\n=== Test 2: Destination only in backup system ===")
    response = await agent.run("What flights are available to Berlin?")
    print(response)

    return str(response)


async def run_self_evaluation(prior_response: str) -> None:
    """Separate evaluator agent scores the prior response."""
    print("\n=== Self-Evaluation ===")

    evaluation_agent = chat_client.as_agent(
        tools=[get_flight_times, get_flight_times_backup],
        name="ResponseEvaluator",
        instructions="""You are a quality evaluator for travel agent responses.
Given a travel question and the agent's response, evaluate:
1. Completeness: Did it answer all parts of the question? (1-5)
2. Accuracy: Is the information correct? (1-5)
3. Helpfulness: Would a traveler find this useful? (1-5)
Provide a brief evaluation with scores and one suggestion for improvement.""",
    )

    eval_prompt = f"""Question: What flights are available to Berlin?
Agent Response: {prior_response}

Please evaluate the above response."""

    evaluation = await evaluation_agent.run(eval_prompt)
    print(evaluation)
    print()


async def main() -> None:
    prior_response = await run_flight_booking_tests()
    print("\n" + "-" * 60)
    await run_self_evaluation(prior_response)


if __name__ == "__main__":
    asyncio.run(main())
