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


class DestinationRecommendation(BaseModel):
    destination: str
    available: bool
    best_season: str
    highlights: list[str]
    estimated_budget_usd: int


class TravelRecommendations(BaseModel):
    recommendations: list[DestinationRecommendation]
    personalized_note: str


@tool(approval_mode="never_require")
def get_destination_details(
    destination: Annotated[str, "The destination to look up"],
) -> DestinationRecommendation:
    """Get structured details about a vacation destination."""
    details = {
        "Barcelona": DestinationRecommendation(
            destination="Barcelona",
            available=True,
            best_season="May-Jun",
            highlights=["Beach", "Architecture", "Nightlife"],
            estimated_budget_usd=2000,
        ),
        "Tokyo": DestinationRecommendation(
            destination="Tokyo",
            available=True,
            best_season="Mar-Apr",
            highlights=["Culture", "Food", "Technology"],
            estimated_budget_usd=2500,
        ),
        "Cape Town": DestinationRecommendation(
            destination="Cape Town",
            available=False,
            best_season="Nov-Mar",
            highlights=["Nature", "Wine", "Adventure"],
            estimated_budget_usd=1800,
        ),
    }
    return details.get(
        destination,
        DestinationRecommendation(
            destination=destination,
            available=False,
            best_season="Unknown",
            highlights=[],
            estimated_budget_usd=0,
        ),
    )


async def pattern_clear_instructions() -> None:
    """Pattern 1: Clear agent instructions."""
    print("=== Pattern 1: Clear Agent Instructions ===\n")

    agent = chat_client.as_agent(
        name="TravelConcierge",
        instructions="""You are a luxury travel concierge named Alex. Your role is to:
1. Understand the traveler's preferences (budget, climate, activities)
2. Check destination availability before making recommendations
3. Provide detailed, personalized travel suggestions
4. Always mention visa requirements and best travel seasons
Be warm, professional, and enthusiastic about travel.""",
    )

    response = await agent.run(
        "I'd love a week-long vacation somewhere with great food and history. Budget around $2500."
    )
    print(response)
    print()


async def pattern_structured_output() -> None:
    """Pattern 2: Structured output with Pydantic models."""
    print("=== Pattern 2: Structured Output with Pydantic Models ===\n")

    structured_agent = chat_client.as_agent(
        name="StructuredTravelExpert",
        instructions=(
            "You are a travel expert. Recommend destinations based on traveler "
            "preferences. Use the get_destination_details tool."
        ),
        tools=[get_destination_details],
    )

    response = await structured_agent.run(
        "Recommend 3 destinations for a culture-loving traveler with a $2500 budget",
        options={"response_format": TravelRecommendations},
    )

    if response and response.value:
        result: TravelRecommendations = response.value
        for rec in result.recommendations:
            status = "Available" if rec.available else "Not available"
            print(f"{rec.destination} ({status})")
            print(f"  Best season: {rec.best_season}")
            print(f"  Highlights: {', '.join(rec.highlights)}")
            print(f"  Estimated budget: ${rec.estimated_budget_usd}")
            print()
        print(f"Note: {result.personalized_note}")
    else:
        print("No validated structured response was returned.")
        print(response)
    print()


async def pattern_single_responsibility() -> None:
    """Pattern 3: Single responsibility agents."""
    print("=== Pattern 3: Single Responsibility Agents ===\n")

    destination_agent = chat_client.as_agent(
        name="DestinationExpert",
        tools=[get_destination_details],
        instructions="""You are a destination research specialist. Your only job is to:
1. Evaluate destinations based on traveler preferences
2. Check availability using the provided tool
3. Return a short ranked list with pros/cons
Do NOT discuss flights, hotels, or logistics — another agent handles that.""",
    )

    logistics_agent = chat_client.as_agent(
        name="LogisticsPlanner",
        instructions="""You are a travel logistics planner. Your only job is to:
1. Create a day-by-day itinerary for the chosen destination
2. Suggest flight and hotel options within the stated budget
3. Note visa requirements and travel insurance recommendations
Do NOT recommend destinations — another agent handles that.""",
    )

    dest_response = await destination_agent.run(
        "I want a week of culture and food for under $2500. Where should I go?"
    )
    print("=== Destination Expert ===")
    print(dest_response)

    logistics_response = await logistics_agent.run(
        f"Plan a week-long trip based on this recommendation:\n{dest_response}"
    )
    print("\n=== Logistics Planner ===")
    print(logistics_response)
    print()


async def main() -> None:
    await pattern_clear_instructions()
    print("-" * 60)
    print()
    await pattern_structured_output()
    print("-" * 60)
    print()
    await pattern_single_responsibility()


if __name__ == "__main__":
    asyncio.run(main())
