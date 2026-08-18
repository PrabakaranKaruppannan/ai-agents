import asyncio
import os
import sys

import dotenv
from agent_framework import AgentResponseUpdate, WorkflowBuilder
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

planner_agent = chat_client.as_agent(
    name="TravelPlanner",
    instructions=(
        "You are a travel planning specialist. Create detailed trip itineraries "
        "based on the traveler's preferences. Include daily schedules, must-see "
        "attractions, and logistical tips."
    ),
)

concierge_agent = chat_client.as_agent(
    name="TravelConcierge",
    instructions=(
        "You are a travel concierge who reviews and enhances trip plans. Review "
        "the plan for completeness, add local insider tips, suggest restaurants, "
        "and identify potential issues. Provide your feedback in a constructive format."
    ),
)

budget_agent = chat_client.as_agent(
    name="BudgetReviewer",
    instructions=(
        "You are a budget-conscious travel advisor. Review the proposed trip plan "
        "and concierge enhancements against the traveler's stated budget. Estimate "
        "costs for flights, hotels, meals, and activities. Flag anything that risks "
        "exceeding the budget and suggest cost-saving alternatives while preserving "
        "the trip's quality."
    ),
)

PROMPT = (
    "Plan a 5-day trip to Paris for a food-loving couple on a $3000 budget."
)


async def stream_workflow(workflow, prompt: str) -> None:
    last_author = None
    events = workflow.run(prompt, stream=True)
    async for event in events:
        if event.type == "output" and isinstance(event.data, AgentResponseUpdate):
            update = event.data
            author = update.author_name
            if author != last_author:
                if last_author is not None:
                    print()
                print(f"\n{'=' * 50}")
                print(f"{author}:")
                print(f"{'=' * 50}")
                last_author = author
            print(update.text, end="", flush=True)
    print()


async def two_agent_workflow() -> None:
    print("=== Sequential Workflow: Planner → Concierge ===\n")
    workflow = (
        WorkflowBuilder(start_executor=planner_agent)
        .add_edge(planner_agent, concierge_agent)
        .build()
    )
    await stream_workflow(workflow, PROMPT)


async def three_agent_workflow() -> None:
    print("\n=== Extended Workflow: Planner → Concierge → BudgetReviewer ===\n")
    extended_workflow = (
        WorkflowBuilder(start_executor=planner_agent)
        .add_edge(planner_agent, concierge_agent)
        .add_edge(concierge_agent, budget_agent)
        .build()
    )
    await stream_workflow(extended_workflow, PROMPT)


async def main() -> None:
    await two_agent_workflow()
    print("\n" + "-" * 60)
    await three_agent_workflow()


if __name__ == "__main__":
    asyncio.run(main())
