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

currency_agent = chat_client.as_agent(
    name="CurrencyExchangeAgent",
    instructions="""You are a currency exchange specialist. You help travelers understand:
- Current exchange rates between currencies
- Best times to exchange money
- Tips for getting the best rates
When asked about a destination, provide relevant currency information.""",
)

activity_agent = chat_client.as_agent(
    name="ActivityPlannerAgent",
    instructions="""You are a local activities specialist. You recommend:
- Must-see attractions and hidden gems
- Local experiences and cultural activities
- Restaurant and dining recommendations
Tailor suggestions to the traveler's interests.""",
)

travel_manager = chat_client.as_agent(
    name="TravelManagerAgent",
    instructions="""You are a travel manager who coordinates between specialist agents.
When planning a trip:
1. Gather currency information from the currency specialist
2. Get activity recommendations from the activity planner
3. Synthesize everything into a cohesive travel brief
Present the final plan in an organized, easy-to-read format.""",
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


async def main() -> None:
    print("=== A2A-Style Multi-Agent Collaboration ===\n")
    print("CurrencyExchangeAgent → ActivityPlannerAgent → TravelManagerAgent\n")

    workflow = (
        WorkflowBuilder(start_executor=currency_agent)
        .add_edge(currency_agent, activity_agent)
        .add_edge(activity_agent, travel_manager)
        .build()
    )

    await stream_workflow(
        workflow,
        "Plan a week-long trip to Tokyo. I love food, temples, and technology.",
    )


if __name__ == "__main__":
    asyncio.run(main())
