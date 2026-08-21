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
def search_accommodations(
    location: Annotated[str, "The city to search for accommodations"],
    check_in: Annotated[str, "Check-in date (YYYY-MM-DD)"],
    check_out: Annotated[str, "Check-out date (YYYY-MM-DD)"],
    guests: Annotated[int, "Number of guests"] = 2,
) -> str:
    """Search for accommodations (simulating an MCP-connected Airbnb tool).

    In production, this would be discovered via MCP from an accommodation service.
    """
    listings = {
        "Tokyo": [
            {"name": "Shinjuku Modern Apartment", "price": 120, "rating": 4.8},
            {"name": "Traditional Ryokan in Asakusa", "price": 200, "rating": 4.9},
            {"name": "Shibuya Studio", "price": 85, "rating": 4.5},
        ],
        "Paris": [
            {"name": "Le Marais Charming Flat", "price": 150, "rating": 4.7},
            {"name": "Montmartre Artist Loft", "price": 110, "rating": 4.6},
        ],
        "Barcelona": [
            {"name": "Gothic Quarter Penthouse", "price": 130, "rating": 4.8},
            {"name": "Barceloneta Beach Flat", "price": 95, "rating": 4.4},
        ],
    }
    results = listings.get(location, [])
    if not results:
        return f"No accommodations found in {location}"
    output = (
        f"Accommodations in {location} "
        f"({check_in} to {check_out}, {guests} guests):\n"
    )
    for listing in results:
        output += (
            f"  - {listing['name']}: ${listing['price']}/night "
            f"(★{listing['rating']})\n"
        )
    return output


@tool(approval_mode="never_require")
def get_local_experiences(
    location: Annotated[str, "The city to find experiences in"],
    interest: Annotated[
        str, "Type of experience (food, culture, adventure, etc.)"
    ] = "all",
) -> str:
    """Get local experiences and activities (simulating an MCP-connected tourism tool)."""
    experiences = {
        "Tokyo": {
            "food": [
                "Tsukiji Market Tour ($45)",
                "Ramen Making Class ($60)",
                "Sake Tasting ($35)",
            ],
            "culture": [
                "Tea Ceremony ($50)",
                "Samurai Museum ($15)",
                "Sumo Tournament ($80)",
            ],
            "adventure": [
                "Mt. Fuji Day Trip ($120)",
                "Go-kart City Tour ($80)",
            ],
        },
        "Paris": {
            "food": [
                "Wine & Cheese Tasting ($55)",
                "Cooking Class ($90)",
                "Market Tour ($40)",
            ],
            "culture": [
                "Louvre Guided Tour ($35)",
                "Montmartre Art Walk ($25)",
            ],
        },
    }
    city_exp = experiences.get(location, {})
    if not city_exp:
        return f"No experiences found in {location}"
    if interest != "all" and interest in city_exp:
        items = city_exp[interest]
        return (
            f"{interest.title()} experiences in {location}:\n"
            + "\n".join(f"  - {e}" for e in items)
        )
    output = f"All experiences in {location}:\n"
    for cat, items in city_exp.items():
        output += f"\n  {cat.title()}:\n"
        for item in items:
            output += f"    - {item}\n"
    return output


async def main() -> None:
    print("=== MCP-Style Tool Discovery Agent ===\n")

    agent = chat_client.as_agent(
        tools=[search_accommodations, get_local_experiences],
        name="AccommodationAgent",
        instructions="""You are an accommodation and travel experiences specialist powered by MCP-connected services.

Help travelers find the perfect place to stay and things to do. When searching:
1. Use the search_accommodations tool to find listings
2. Use the get_local_experiences tool to suggest activities
3. Compare options and make personalized recommendations
4. Consider the traveler's budget, interests, and travel style""",
    )

    response = await agent.run(
        "I'm visiting Tokyo for 5 nights in April with my partner. "
        "We love traditional Japanese culture and food. "
        "Find us a place to stay and suggest some experiences."
    )
    print(response)
    print()


if __name__ == "__main__":
    asyncio.run(main())
