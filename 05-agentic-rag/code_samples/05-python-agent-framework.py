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

TRAVEL_KNOWLEDGE_BASE = {
    "Barcelona": (
        "Barcelona is Spain's cosmopolitan capital of Catalonia. "
        "Best visited Mar-May or Sep-Nov. Known for Gaudí architecture, "
        "La Rambla, beaches. Average daily cost: $150-200."
    ),
    "Tokyo": (
        "Tokyo is Japan's capital, mixing ultramodern with traditional. "
        "Best visited Mar-Apr (cherry blossoms) or Oct-Nov. Known for "
        "Shibuya, temples, sushi. Average daily cost: $200-250."
    ),
    "Paris": (
        "Paris is France's capital and a global center for art, fashion, "
        "and culture. Best visited Apr-Jun or Sep-Oct. Known for Eiffel "
        "Tower, Louvre, cuisine. Average daily cost: $180-250."
    ),
    "Cape Town": (
        "Cape Town sits on South Africa's southwest tip. Best visited "
        "Nov-Mar. Known for Table Mountain, wine regions, wildlife. "
        "Average daily cost: $100-150."
    ),
}


@tool(approval_mode="never_require")
def search_travel_knowledge(
    query: Annotated[str, "The search query about a travel destination"],
) -> str:
    """Search the travel knowledge base for destination information."""
    results = []
    for destination, info in TRAVEL_KNOWLEDGE_BASE.items():
        if query.lower() in destination.lower() or any(
            word in info.lower() for word in query.lower().split()
        ):
            results.append(f"**{destination}**: {info}")
    return (
        "\n\n".join(results)
        if results
        else "No matching destinations found in the knowledge base."
    )


async def basic_rag_agent() -> None:
    """Agent that retrieves from the knowledge base before answering."""
    print("=== Agentic RAG — Travel Advisor ===\n")

    agent = chat_client.as_agent(
        tools=[search_travel_knowledge],
        name="TravelRAGAgent",
        instructions="""You are a knowledgeable travel advisor. Before answering questions about destinations:
1. ALWAYS search the travel knowledge base first
2. Base your answers on retrieved information
3. If information is not in the knowledge base, say so clearly
4. Provide specific details like costs, best seasons, and highlights.""",
    )

    response = await agent.run(
        "I'm interested in visiting somewhere with great architecture. "
        "What destinations would you recommend?"
    )
    print(response)
    print()


async def maker_checker_rag() -> None:
    """Iterative retrieval — search, verify, then recommend."""
    print("=== Iterative Retrieval — Maker-Checker ===\n")

    checker_agent = chat_client.as_agent(
        tools=[search_travel_knowledge],
        name="TravelRAGCheckerAgent",
        instructions="""You are a meticulous travel advisor who double-checks recommendations.
When answering travel questions:
1. Search for relevant destinations first
2. For each destination found, search again with the destination name to get full details
3. Compare the options using verified information
4. Present a final recommendation with specific costs, best travel times, and highlights
5. If any detail seems incomplete, search once more to confirm before responding.""",
    )

    response = await checker_agent.run(
        "I have a $175/day budget and want to travel in April. "
        "Which destinations fit my budget and timing?"
    )
    print(response)
    print()


async def main() -> None:
    await basic_rag_agent()
    print("-" * 60)
    print()
    await maker_checker_rag()


if __name__ == "__main__":
    asyncio.run(main())
