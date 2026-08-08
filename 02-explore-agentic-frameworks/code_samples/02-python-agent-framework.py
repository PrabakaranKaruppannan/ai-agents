import asyncio
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Annotated

import dotenv
from agent_framework import tool
from agent_framework.foundry import FoundryChatClient
from azure.identity import AzureCliCredential

# Avoid Windows console UnicodeEncodeError when streaming responses.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def ensure_azure_cli_on_path() -> None:
    """AzureCliCredential needs `az` on PATH. Fresh installs often aren't visible in Git Bash."""
    if shutil.which("az"):
        return

    candidates = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        / "Microsoft SDKs"
        / "Azure"
        / "CLI2"
        / "wbin",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Microsoft SDKs"
        / "Azure"
        / "CLI2"
        / "wbin",
    ]
    for folder in candidates:
        if (folder / "az.cmd").exists() or (folder / "az.exe").exists():
            os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")
            return

    raise RuntimeError(
        "Azure CLI was not found on PATH. Install it, then restart the terminal, "
        "or add this folder to PATH: "
        r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin"
    )


logging.getLogger("agent_framework.foundry").setLevel(logging.ERROR)

dotenv.load_dotenv(dotenv.find_dotenv())
ensure_azure_cli_on_path()

endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
model = os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME")

if not endpoint or not model:
    raise ValueError(
        "Missing required environment variables. "
        "Please set AZURE_AI_PROJECT_ENDPOINT and AZURE_AI_MODEL_DEPLOYMENT_NAME in your .env file."
    )

provider = FoundryChatClient(
    project_endpoint=endpoint,
    model=model,
    credential=AzureCliCredential(),
)


@tool(approval_mode="never_require")
def check_destination_availability(
    destination: Annotated[str, "The destination to check availability for"],
) -> str:
    """Check if a vacation destination is currently available for booking."""
    available = {
        "Barcelona": True,
        "Tokyo": True,
        "Cape Town": False,
        "Vancouver": True,
        "Dubai": False,
    }
    is_available = available.get(destination, False)
    return f"{destination} is {'available' if is_available else 'not available'} for booking."


agent = provider.as_agent(
    name="TravelAvailabilityAgent",
    instructions=(
        "You are a travel booking agent. Help users check destination availability "
        "and make recommendations. Always check availability before recommending a destination."
    ),
    tools=[check_destination_availability],
)


async def main() -> None:
    session = agent.create_session()

    response = await agent.run(
        "Which destinations do you have available?",
        session=session,
    )
    print(f"Agent: {response}")

    print()

    response = await agent.run(
        "I'd like to go somewhere warm. What's available?",
        session=session,
    )
    print(f"Agent: {response}")


if __name__ == "__main__":
    asyncio.run(main())
