import asyncio
import os
import sys
from typing import Annotated

import dotenv
from agent_framework import tool
from agent_framework.openai import OpenAIChatClient
from pydantic import BaseModel

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


class TravelSubTask(BaseModel):
    task_id: int
    description: str
    assigned_agent: str  # "flight_agent", "hotel_agent", "activity_agent"
    priority: str  # "high", "medium", "low"
    dependencies: list[int] = []


class TravelPlan(BaseModel):
    destination: str
    trip_duration_days: int
    subtasks: list[TravelSubTask]
    total_estimated_budget_usd: int
    notes: str


@tool(approval_mode="never_require")
def book_flight(
    destination: Annotated[str, "The destination city"],
    departure_date: Annotated[str, "Departure date (YYYY-MM-DD)"],
    return_date: Annotated[str, "Return date (YYYY-MM-DD)"],
) -> str:
    """Search and book flights for the trip."""
    return (
        f"Flight booked to {destination}: {departure_date} → {return_date}, "
        f"confirmation #FLT-{hash(destination) % 10000:04d}"
    )


@tool(approval_mode="never_require")
def reserve_hotel(
    city: Annotated[str, "The city for the hotel"],
    check_in: Annotated[str, "Check-in date (YYYY-MM-DD)"],
    check_out: Annotated[str, "Check-out date (YYYY-MM-DD)"],
    guests: Annotated[int, "Number of guests"],
) -> str:
    """Reserve a hotel room in the destination city."""
    return (
        f"Hotel reserved in {city}: {check_in} to {check_out} for {guests} guests, "
        f"confirmation #HTL-{hash(city) % 10000:04d}"
    )


@tool(approval_mode="never_require")
def book_activity(
    activity_name: Annotated[str, "Name of the activity or tour"],
    date: Annotated[str, "Date of the activity (YYYY-MM-DD)"],
    participants: Annotated[int, "Number of participants"],
) -> str:
    """Book a tour, museum visit, or other activity."""
    return (
        f"Activity booked: {activity_name} on {date} for {participants} people, "
        f"confirmation #ACT-{hash(activity_name) % 10000:04d}"
    )


async def create_travel_plan() -> TravelPlan | None:
    """Front desk planner: decompose a request into a structured TravelPlan."""
    print("=== Planning Agent — Task Decomposition ===\n")

    planning_agent = chat_client.as_agent(
        name="TravelPlanner",
        instructions="""You are a travel planning agent. When given a travel request:
1. Break it into specific subtasks (flights, hotels, activities, logistics)
2. Assign each subtask to the appropriate specialist agent
3. Set priorities and identify dependencies between tasks
4. Estimate the total budget""",
    )

    result = await planning_agent.run(
        "Plan a 7-day trip to Paris for a couple interested in art, cuisine, "
        "and history. Budget around $5000.",
        options={"response_format": TravelPlan},
    )

    if not result or not result.value:
        print("No validated travel plan was returned.")
        print(result)
        return None

    plan: TravelPlan = result.value
    print(f"Destination: {plan.destination}")
    print(f"Duration: {plan.trip_duration_days} days")
    print(f"Budget: ${plan.total_estimated_budget_usd}")
    print("\nSubtasks:")
    for task in plan.subtasks:
        print(
            f"  [{task.priority}] {task.task_id}. {task.description} "
            f"→ {task.assigned_agent}"
        )
    print()
    return plan


async def execute_travel_plan(plan: TravelPlan) -> None:
    """Concierge: execute the plan with specialist booking tools."""
    print("=== Concierge Agent — Plan Execution ===\n")

    concierge_agent = chat_client.as_agent(
        name="Concierge",
        instructions="""You are a travel concierge executing a structured travel plan.
Use the available tools to fulfil each subtask. Work through the subtasks in order,
respecting dependencies. Summarise the results when finished.""",
        tools=[book_flight, reserve_hotel, book_activity],
    )

    subtask_lines = "\n".join(
        f"- [{t.priority}] {t.task_id}. {t.description} "
        f"(agent: {t.assigned_agent}, deps: {t.dependencies})"
        for t in plan.subtasks
    )
    execution_prompt = (
        f"Execute the following travel plan for {plan.destination} "
        f"({plan.trip_duration_days} days, ${plan.total_estimated_budget_usd} budget):\n"
        f"{subtask_lines}"
    )

    exec_response = await concierge_agent.run(execution_prompt)
    print(exec_response)
    print()


async def main() -> None:
    plan = await create_travel_plan()
    if plan is None:
        return

    print("-" * 60)
    print()
    await execute_travel_plan(plan)


if __name__ == "__main__":
    asyncio.run(main())
