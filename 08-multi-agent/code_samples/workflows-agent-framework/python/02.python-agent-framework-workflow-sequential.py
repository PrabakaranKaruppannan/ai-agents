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

sales_agent = chat_client.as_agent(
    name="Sales-Agent",
    instructions=(
        "You are my furniture sales consultant. From the room description, "
        "identify furniture pieces and give a purchase suggestion for each."
    ),
)

price_agent = chat_client.as_agent(
    name="Price-Agent",
    instructions="""You are a furniture pricing specialist and budget consultant. Your responsibilities include:
1. Analyze furniture items and provide realistic price ranges based on quality, brand, and market standards
2. Break down pricing by individual furniture pieces
3. Provide budget-friendly alternatives and premium options
4. Consider different price tiers (budget, mid-range, premium)
5. Include estimated total costs for room setups
6. Suggest where to find the best deals and shopping recommendations
7. Factor in additional costs like delivery, assembly, and accessories
8. Provide seasonal pricing insights and best times to buy
Always format your response with clear price breakdowns and explanations for the pricing rationale.""",
)

quote_agent = chat_client.as_agent(
    name="Quote-Agent",
    instructions="""You are an assistant that creates a quote for furniture purchase.
1. Create a well-structured quote document that includes:
2. A title page with the document title, date, and client name
3. An introduction summarizing the purpose of the document
4. A summary section with total estimated costs and recommendations
5. Use clear headings, bullet points, and tables for easy readability
6. All quotes are presented in markdown form""",
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
        WorkflowBuilder(start_executor=sales_agent)
        .add_edge(sales_agent, price_agent)
        .add_edge(price_agent, quote_agent)
        .build()
    )

    print_workflow_viz(workflow)

    message = (
        "I am furnishing a modern living room and want pieces that fit a warm, "
        "inviting style: a comfortable three-seat sofa, two accent armchairs, a "
        "wooden coffee table, a TV stand, a floor lamp, and a soft area rug. "
        "Please find appropriate furniture and give the corresponding price for "
        "each piece, then produce a final purchase quote."
    )

    events = await workflow.run(message)
    outputs = events.get_outputs()
    result = outputs[0].text if outputs else ""
    print("\n=== Workflow Result ===\n")
    print(result.replace("None", ""))


if __name__ == "__main__":
    asyncio.run(main())
