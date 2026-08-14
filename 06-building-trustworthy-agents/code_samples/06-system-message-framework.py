import os
import sys

import dotenv
from openai import OpenAI

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

# Azure OpenAI Responses API via the stable /openai/v1/ endpoint.
client = OpenAI(
    base_url=f"{endpoint.rstrip('/')}/openai/v1/",
    api_key=api_key,
)

role = "travel agent"
company = "contoso travel"
responsibility = "booking flights"


def main() -> None:
    response = client.responses.create(
        model=deployment,
        input=[
            {
                "role": "system",
                "content": (
                    "You are an expert at creating AI agent assistants. "
                    "You will be provided a company name, role, responsibilities and other "
                    "information that you will use to provide a system prompt for. "
                    "To create the system prompt, be descriptive as possible and provide a "
                    "structure that a system using an LLM can better understand the role and "
                    "responsibilities of the AI assistant."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"You are {role} at {company} that is responsible for {responsibility}."
                ),
            },
        ],
        temperature=1.0,
        max_output_tokens=1000,
        top_p=1.0,
        store=False,
    )
    print(response.output_text)


if __name__ == "__main__":
    main()
