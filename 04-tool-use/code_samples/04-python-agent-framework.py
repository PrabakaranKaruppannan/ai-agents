import asyncio
import os
import sys
from typing import Annotated

import dotenv
from agent_framework import tool
from agent_framework.openai import OpenAIChatClient
from pydantic import BaseModel

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
def get_destinations() -> list[str]:
    """Get available vacation destinations."""
    return ["Barcelona", "Paris", "Berlin", "Tokyo", "Sydney", "New York City"]


@tool(approval_mode="never_require")
def check_availability(
    destination: Annotated[str, "The destination to check"],
) -> str:
    """Check booking availability for a destination."""
    availability = {
        "Barcelona": "Available - 3 spots left",
        "Paris": "Available",
        "Berlin": "Sold out",
        "Tokyo": "Available - 1 spot left",
        "Sydney": "Available",
        "New York City": "Available",
    }
    return availability.get(destination, "Unknown destination")


@tool(approval_mode="never_require")
def get_flight_info(
    origin: Annotated[str, "Origin airport code"],
    destination: Annotated[str, "Destination airport code"],
) -> str:
    """Get flight information between two cities."""
    flights = {
        "LHR-BCN": "BA 2042, Departs 08:30, Arrives 11:45, $350",
        "LHR-CDG": "AF 1081, Departs 09:15, Arrives 11:30, $280",
        "LHR-NRT": "JL 044, Departs 11:00, Arrives 07:00+1, $890",
    }
    return flights.get(
        f"{origin}-{destination}",
        f"No direct flights from {origin} to {destination}",
    )


@tool(approval_mode="always_require")
def book_flight(
    origin: Annotated[str, "Origin airport code"],
    destination: Annotated[str, "Destination airport code"],
    passenger_name: Annotated[str, "Full name of the passenger"],
) -> str:
    """Book a flight for a passenger. Requires approval before executing."""
    return (
        f"Flight booked from {origin} to {destination} "
        f"for {passenger_name}. Confirmation #TRV-2024-{hash(passenger_name) % 10000:04d}"
    )


class BookingRecommendation(BaseModel):
    destination: str
    available: bool
    flight_details: str
    estimated_cost: int


class TravelPlan(BaseModel):
    recommendations: list[BookingRecommendation]


async def multi_tool_agent() -> None:
    """Compose multiple tools so the agent can answer complex queries."""
    print("=== Multi-Tool Travel Agent ===\n")

    travel_tools = [get_destinations, check_availability, get_flight_info]
    agent = chat_client.as_agent(
        name="TravelToolAgent",
        instructions=(
            "You are a travel agent. Use the available tools to answer questions "
            "about destinations, availability, and flights."
        ),
        tools=travel_tools,
    )

    response = await agent.run(
        "What destinations do you have? Which ones are still available?"
    )
    print(response)
    print()


async def structured_output_with_tools() -> None:
    """Force structured results via response_format + tools."""
    print("=== Structured Output with Tools ===\n")

    structured_agent = chat_client.as_agent(
        name="StructuredTravelAgent",
        instructions=(
            "You are a travel agent. Use the available tools to find destinations, "
            "check availability, and get flight info. Return structured results."
        ),
        tools=[get_destinations, check_availability, get_flight_info],
    )

    response = await structured_agent.run(
        "I want to fly from London Heathrow to somewhere warm in Europe. "
        "Check what's available.",
        options={"response_format": TravelPlan},
    )

    if response and response.value:
        plan: TravelPlan = response.value
        for rec in plan.recommendations:
            status = "Available" if rec.available else "Not available"
            print(f"{rec.destination} ({status})")
            print(f"  Flight: {rec.flight_details}")
            print(f"  Estimated cost: ${rec.estimated_cost}")
            print()
    elif response:
        print(response)
    print()


def tool_approval_demo() -> None:
    """Show approval_mode metadata for a side-effect tool."""
    print("=== Tool Approval Patterns ===\n")
    print("Tool name:", book_flight.name)
    print("Approval mode:", book_flight.approval_mode)
    print()


async def main() -> None:
    await multi_tool_agent()
    print("-" * 60)
    print()
    await structured_output_with_tools()
    print("-" * 60)
    print()
    tool_approval_demo()


if __name__ == "__main__":
    asyncio.run(main())
