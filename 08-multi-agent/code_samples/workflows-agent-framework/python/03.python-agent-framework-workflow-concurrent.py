import asyncio
import os
import sys

import dotenv
from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowViz,
    handler,
)
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

research_agent = chat_client.as_agent(
    name="Researcher-Agent",
    instructions=(
        "You are my travel researcher, working with me to analyze the destination, "
        "list relevant attractions, and make detailed plans for each attraction."
    ),
)

plan_agent = chat_client.as_agent(
    name="Plan-Agent",
    instructions=(
        "You are my travel planner, working with me to create a detailed travel "
        "plan based on the researcher's findings."
    ),
)


class InputDispatcher(Executor):
    """Forward the user input unchanged to all participating agents."""

    @handler
    async def forward(self, text: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(text)


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
    dispatcher = InputDispatcher(id="dispatcher")
    agents = [research_agent, plan_agent]

    workflow = (
        WorkflowBuilder(
            start_executor=dispatcher,
            output_executors=agents,
        )
        .add_fan_out_edges(dispatcher, agents)
        .build()
    )

    print_workflow_viz(workflow)

    events = await workflow.run("Plan a trip to Seattle in December")
    outputs = events.get_outputs()

    if outputs:
        print("===== Final Aggregated Responses =====")
        for i, response in enumerate(outputs, start=1):
            print(f"{'-' * 60}\n\n{i:02d}:\n{response.text}")


if __name__ == "__main__":
    asyncio.run(main())
