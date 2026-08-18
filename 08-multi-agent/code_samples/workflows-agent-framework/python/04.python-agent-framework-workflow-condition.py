"""Conditional workflow: draft → review → publish (or reject).

Adapted from the notebook to use OpenAIChatClient + API key auth.
The original notebook used Azure AI Agent Client with Bing and Code Interpreter;
this version keeps the conditional routing pattern with local file publishing.
"""

import asyncio
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import dotenv
from agent_framework import (
    AgentExecutor,
    AgentExecutorResponse,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowViz,
    executor,
)
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

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

EvangelistInstructions = """
You are a technology evangelist creating a first draft for a technical tutorial.
1. Expand each knowledge point in the outline with clear explanations.
2. Each knowledge point must be explained in detail.
3. Rewrite the content including title, outline, and corresponding content.
4. The content must be more than 200 words.
5. Output draft as Markdown format.
6. Return JSON with field 'draft_content' (string).
Do not create sample code.
"""

ContentReviewerInstructions = """
You are a content reviewer for a publishing company. Check whether the tutorial draft meets requirements:
1. If draft content is less than 200 words, set 'review_result' to 'No' and 'reason' to 'Content is too short'.
   If more than 200 words, set 'review_result' to 'Yes' and 'reason' to 'The content is good'.
2. Set 'draft_content' to the original draft content.
3. Return JSON with fields 'review_result' (Yes or No), 'reason' (string), and 'draft_content' (string).
"""

OUTLINE_CONTENT = """
# Introduce AI Agent

## What's AI Agent
https://github.com/microsoft/ai-agents-for-beginners/tree/main/01-intro-to-ai-agents
***Note*** Don't create any sample code

## Introduce Microsoft Foundry Agent Service
https://learn.microsoft.com/en-us/azure/ai-foundry/agents/overview
***Note*** Don't create any sample code

## Microsoft Agent Framework
https://github.com/microsoft/agent-framework
***Note*** Don't create any sample code
"""


class EvangelistAgent(BaseModel):
    draft_content: str


class ReviewAgent(BaseModel):
    review_result: Literal["Yes", "No"]
    reason: str
    draft_content: str


@dataclass
class ReviewResult:
    review_result: str
    reason: str
    draft_content: str


@executor(id="to_reviewer_result")
async def to_reviewer_result(
    response: AgentExecutorResponse, ctx: WorkflowContext[ReviewResult]
) -> None:
    text = response.agent_run_response.text
    print(f"Raw response from reviewer agent: {text}")
    # Prefer structured value when available; fall back to JSON text.
    value = getattr(response.agent_run_response, "value", None)
    if isinstance(value, ReviewAgent):
        parsed = value
    else:
        parsed = ReviewAgent.model_validate_json(text)
    await ctx.send_message(
        ReviewResult(
            review_result=parsed.review_result,
            reason=parsed.reason,
            draft_content=parsed.draft_content,
        )
    )


def select_targets(review: ReviewResult, target_ids: list[str]) -> list[str]:
    handle_review_id, save_draft_id = target_ids
    if review.review_result == "Yes":
        return [save_draft_id]
    return [handle_review_id]


@executor(id="handle_review")
async def handle_review(review: ReviewResult, ctx: WorkflowContext[str]) -> None:
    await ctx.yield_output(
        f"Review failed: {review.reason}, please revise the draft."
    )


@executor(id="save_draft")
async def save_draft(review: ReviewResult, ctx: WorkflowContext[str]) -> None:
    filename = datetime.now().strftime("%Y%m%d%H%M%S") + ".md"
    path = OUTPUT_DIR / filename
    path.write_text(review.draft_content, encoding="utf-8")
    await ctx.yield_output(
        f"Review approved ({review.reason}). Published draft to {path}"
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
    evangelist_agent = AgentExecutor(
        chat_client.as_agent(
            name="Evangelist",
            instructions=EvangelistInstructions,
            default_options={"response_format": EvangelistAgent},
        ),
        id="evangelist_agent",
    )
    reviewer_agent = AgentExecutor(
        chat_client.as_agent(
            name="Reviewer",
            instructions=ContentReviewerInstructions,
            default_options={"response_format": ReviewAgent},
        ),
        id="reviewer_agent",
    )

    workflow = (
        WorkflowBuilder(start_executor=evangelist_agent)
        .add_edge(evangelist_agent, reviewer_agent)
        .add_edge(reviewer_agent, to_reviewer_result)
        .add_multi_selection_edge_group(
            to_reviewer_result,
            [handle_review, save_draft],
            selection_func=select_targets,
        )
        .build()
    )

    print_workflow_viz(workflow)

    task = (
        "You are an evangelist. Write a draft based on the following outline. "
        "After the draft is created, the reviewer checks it; if it meets the "
        "requirements it is saved as a Markdown file, otherwise report that "
        "the draft must be revised.\n\n"
        "The provided outline content and related links is as follows:\n"
        + OUTLINE_CONTENT
    )

    events = await workflow.run(task)
    outputs = events.get_outputs()
    print("\n=== Workflow Outputs ===\n")
    if outputs:
        for i, output in enumerate(outputs, start=1):
            text = getattr(output, "text", None) or str(output)
            print(f"{i:02d}: {text}")
    else:
        print("(no outputs)")


if __name__ == "__main__":
    asyncio.run(main())
