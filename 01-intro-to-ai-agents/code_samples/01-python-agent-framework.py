import asyncio
import logging
import os
import shutil
import sys
from pathlib import Path

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
def get_destinations() -> list[str]:
    """Get a list of popular vacation destinations."""
    return [
        "Barcelona",
        "Paris",
        "Berlin",
        "Tokyo",
        "Sydney",
        "New York City",
        "Cairo",
        "Cape Town",
        "Rio de Janeiro",
        "Bali",
    ]


agent = provider.as_agent(
    name="TravelAgent",
    instructions=(
        "You are a helpful travel agent. Help users find their perfect vacation "
        "destination based on their preferences. Use the get_destinations tool "
        "to see available destinations."
    ),
    tools=[get_destinations],
)


async def main() -> None:
    response = await agent.run(
        "I'm looking for a warm beach destination. What do you recommend?"
    )
    print(response)

    print("\n--- Streaming ---\n")
    async for chunk in agent.run(
        "Tell me about Tokyo as a travel destination", stream=True
    ):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
