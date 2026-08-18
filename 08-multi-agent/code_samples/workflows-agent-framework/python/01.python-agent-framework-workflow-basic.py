import asyncio
import os
import sys

import dotenv
from agent_framework import WorkflowBuilder, WorkflowViz
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

REVIEWER_NAME = "Concierge"
REVIEWER_INSTRUCTIONS = """
You are a hotel concierge who has opinions about providing the most local and authentic experiences for travelers.
The goal is to determine if the front desk travel agent has recommended the best non-touristy experience for a traveler.
If so, state that it is approved.
If not, provide insight on how to refine the recommendation without using a specific example.
"""

FRONTDESK_NAME = "FrontDesk"
FRONTDESK_INSTRUCTIONS = """
You are a Front Desk Travel Agent with ten years of experience and are known for brevity as you deal with many customers.
The goal is to provide the best activities and locations for a traveler to visit.
Only provide a single recommendation per response.
You're laser focused on the goal at hand.
Don't waste time with chit chat.
Consider suggestions when refining an idea.
"""

reviewer_agent = chat_client.as_agent(
    name=REVIEWER_NAME,
    instructions=REVIEWER_INSTRUCTIONS,
)

front_desk_agent = chat_client.as_agent(
    name=FRONTDESK_NAME,
    instructions=FRONTDESK_INSTRUCTIONS,
)


def print_workflow_viz(workflow) -> None:
    print("Generating workflow visualization...")
    viz = WorkflowViz(workflow)
    print("Mermaid string:\n=======")
    print(viz.to_mermaid())
    print("=======")
    print("DiGraph string:\n=======")
    print(viz.to_digraph())
    print("=======")
    try:
        svg_file = viz.export(format="svg")
        print(f"SVG file saved to: {svg_file}")
    except ImportError as e:
        print(f"SVG export skipped (install graphviz to enable): {e}")


async def main() -> None:
    workflow = (
        WorkflowBuilder(start_executor=front_desk_agent)
        .add_edge(front_desk_agent, reviewer_agent)
        .build()
    )

    print_workflow_viz(workflow)

    events = await workflow.run("I would like to go to Paris.")
    outputs = events.get_outputs()
    result = outputs[0].text if outputs else ""
    print("\n=== Workflow Result ===\n")
    print(result.replace("None", ""))


if __name__ == "__main__":
    asyncio.run(main())
