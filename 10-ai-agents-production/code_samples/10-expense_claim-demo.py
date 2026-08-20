import asyncio
import base64
import os
import sys
from pathlib import Path
from typing import Annotated, List

import dotenv
from agent_framework import AgentResponseUpdate, WorkflowBuilder, tool
from agent_framework.openai import OpenAIChatClient
from pydantic import BaseModel, Field

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

RECEIPT_PATH = Path(__file__).resolve().parent / "receipt.jpg"


class Expense(BaseModel):
    date: str = Field(..., description="Date of expense in dd-MMM-yyyy format")
    description: str = Field(..., description="Expense description")
    amount: float = Field(..., description="Expense amount")
    category: str = Field(
        ...,
        description=(
            "Expense category (e.g., Transportation, Meals, "
            "Accommodation, Miscellaneous)"
        ),
    )


class ExpenseFormatter(BaseModel):
    raw_query: str = Field(..., description="Raw query input containing expense details")

    def parse_expenses(self) -> List[Expense]:
        """Parse 'date|description|amount|category' entries separated by semicolons."""
        expense_list = []
        for expense_str in self.raw_query.split(";"):
            if expense_str.strip():
                parts = expense_str.strip().split("|")
                if len(parts) == 4:
                    date, description, amount, category = parts
                    try:
                        expense_list.append(
                            Expense(
                                date=date.strip(),
                                description=description.strip(),
                                amount=float(amount.strip()),
                                category=category.strip(),
                            )
                        )
                    except ValueError as e:
                        print(f"[LOG] Parse Error: Invalid data in '{expense_str}': {e}")
        return expense_list


@tool(approval_mode="never_require")
def generate_expense_email(
    expense_data: Annotated[
        str,
        "Semicolon-separated expense entries in 'date|description|amount|category' format",
    ],
) -> str:
    """Generate an email to submit an expense claim to the Finance Team."""
    formatter = ExpenseFormatter(raw_query=expense_data)
    expenses = formatter.parse_expenses()
    if not expenses:
        return "No valid expenses found to include in the email."
    total_amount = sum(e.amount for e in expenses)
    email_body = "Dear Finance Team,\n\n"
    email_body += "Please find below the details of my expense claim:\n\n"
    for e in expenses:
        email_body += f"- {e.date} | {e.description}: ${e.amount:.2f} ({e.category})\n"
    email_body += f"\nTotal Amount: ${total_amount:.2f}\n\n"
    email_body += "Receipts for all expenses are attached for your reference.\n\n"
    email_body += "Thank you,\n[Your Name]"
    return email_body


@tool(approval_mode="never_require")
def load_receipt_image(
    image_path: Annotated[str, "Path to the receipt image file"] = "",
) -> str:
    """Load a receipt image and return its base64-encoded data URI for OCR extraction."""
    path = Path(image_path) if image_path else RECEIPT_PATH
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    try:
        with open(path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{image_data}"
    except Exception as e:
        error_msg = f"[LOG] Error loading image '{path}': {str(e)}"
        print(error_msg)
        return error_msg


ocr_agent = chat_client.as_agent(
    tools=[load_receipt_image],
    name="OCRAgent",
    instructions=(
        "You are an expert OCR assistant specialized in extracting structured data "
        "from receipt images. Use the 'load_receipt_image' tool to load the receipt "
        "image, then analyze it and extract travel-related expense details in the "
        "format: 'date|description|amount|category' separated by semicolons. "
        "Follow these rules: "
        "- Date: Convert dates (e.g., '4/4/22') to 'dd-MMM-yyyy' (e.g., '04-Apr-2022'). "
        "- Description: Extract item names. "
        "- Amount: Use numeric values (e.g., '4.50' from '$4.50'). "
        "- Category: Infer from context (e.g., 'Meals' for food, 'Transportation' for "
        "travel, 'Accommodation' for lodging, 'Miscellaneous' otherwise). "
        "Ignore totals, subtotals, or service charges unless they are itemized expenses. "
        "If no expenses are found, return 'No expenses detected'. "
        "Return only the structured data, no additional text."
    ),
)

email_agent = chat_client.as_agent(
    name="EmailAgent",
    tools=[generate_expense_email],
    instructions=(
        "You are an expense claim email generator. Take the travel expense data from "
        "the previous agent (in 'date|description|amount|category' format separated by "
        "semicolons) and use the 'generate_expense_email' tool to produce a professional "
        "expense claim email. Pass the semicolon-separated expense data directly to the tool."
    ),
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
                print(f"# Agent - {author}:")
                print(f"{'=' * 50}")
                last_author = author
            print(update.text, end="", flush=True)
    print()


async def main() -> None:
    if not RECEIPT_PATH.exists():
        raise FileNotFoundError(
            f"Receipt image not found at {RECEIPT_PATH}. "
            "Ensure receipt.jpg is in the code_samples folder."
        )

    # Note: Passing a large base64 receipt as text may exceed context or not be
    # treated as an image by the model. Prefer Azure AI Vision OCR in production.
    workflow = (
        WorkflowBuilder(start_executor=ocr_agent)
        .add_edge(ocr_agent, email_agent)
        .build()
    )

    prompt = (
        f"Please extract the raw text from the receipt image at '{RECEIPT_PATH}', "
        "focusing on travel expenses like dates, descriptions, amounts, and categories "
        "(e.g., Transportation, Accommodation, Meals, Miscellaneous). "
        "Then generate a professional expense claim email."
    )

    await stream_workflow(workflow, prompt)


if __name__ == "__main__":
    asyncio.run(main())
