import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import dotenv
from openai import OpenAI

# Avoid Windows console UnicodeEncodeError when printing responses.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

dotenv.load_dotenv(dotenv.find_dotenv())

DEMO_MODE = True  # set False to use real input() prompts

# Per-run unique log filename so demo runs don't overwrite each other.
GATE_LOG_PATH = Path(
    f"gate_log_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
)

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

LOW_RISK_KEYWORDS = {
    "look",
    "lookup",
    "search",
    "fetch",
    "read",
    "query",
    "view",
    "get",
    "list",
    "weather",
    "summarize",
}
HIGH_RISK_KEYWORDS = {
    "send",
    "email",
    "post",
    "publish",
    "charge",
    "pay",
    "transfer",
    "delete",
    "drop",
    "cancel",
    "refund",
}
MEDIUM_RISK_KEYWORDS = {
    "cache",
    "schedule",
    "reminder",
    "book",
    "reserve",
    "update",
    "increment",
    "log",
}

AUTO_APPROVE_REASONS = {
    "low": "auto-approved (low risk)",
    "medium": "auto-approved (medium risk, queued for batched review)",
}


def gate_action(action_description: str, risk_tier: str, attempt: int = 0) -> dict:
    """Run a single pre-action gate.

    Returns a decision dict with keys: decision, reason, ts.
    Decision is one of: approve, deny, escalate.
    Safe default on EOF or unexpected input is deny.
    """
    print(f"[gate] proposed action ({risk_tier}, attempt={attempt}): {action_description}")

    if DEMO_MODE:
        if risk_tier == "high":
            decision = "approve" if attempt >= 1 else "deny"
            reason = (
                "DEMO_MODE: scripted approval on retry to show loop mechanics"
                if attempt >= 1
                else "DEMO_MODE: high risk denied on first attempt"
            )
        else:
            decision = "approve"
            reason = f"DEMO_MODE canned response for tier={risk_tier}"
    else:
        try:
            raw = input("[gate] approve / deny / escalate? ").strip().lower()
        except EOFError:
            raw = ""
        if raw in {"approve", "deny", "escalate"}:
            decision, reason = raw, "operator input"
        elif raw == "":
            decision, reason = "deny", "no input received, defaulted to deny"
        else:
            decision, reason = "deny", f"invalid input {raw!r}, defaulted to deny"

    return {
        "decision": decision,
        "reason": reason,
        "action": action_description,
        "risk_tier": risk_tier,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def classify_risk(action: str) -> str:
    """Classify an action string into one of: low, medium, high."""
    text = action.lower()
    if any(kw in text for kw in HIGH_RISK_KEYWORDS):
        return "high"
    if any(kw in text for kw in LOW_RISK_KEYWORDS):
        return "low"
    if any(kw in text for kw in MEDIUM_RISK_KEYWORDS):
        return "medium"
    # Fail-safe: unrecognized actions route to batched review.
    return "medium"


def tiered_gate(action: str, attempt: int = 0) -> dict:
    """Classify then gate. Low and medium tiers auto-approve; high blocks."""
    tier = classify_risk(action)
    if tier in AUTO_APPROVE_REASONS:
        return {
            "decision": "approve",
            "reason": AUTO_APPROVE_REASONS[tier],
            "action": action,
            "risk_tier": tier,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
    return gate_action(action, tier, attempt=attempt)


def log_decision(decision: dict) -> None:
    """Append a gate decision to the JSONL audit log."""
    with GATE_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(decision) + "\n")


def propose_action(goal: str, prior_rejection: str | None = None) -> str:
    """Ask the LLM to propose a concrete next action for a goal."""
    system = (
        "You are an action planner for an agent. Propose ONE concrete next\n"
        "action (a single sentence) toward the user's goal. If a prior\n"
        "rejection reason is given, propose a different action that addresses\n"
        "the rejection."
    )
    user_text = f"Goal: {goal}"
    if prior_rejection:
        user_text += f"\n\nPrior proposal was denied. Reason: {prior_rejection}"

    response = client.responses.create(
        model=deployment,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        store=False,
    )
    return response.output_text.strip()


def run_with_revision(goal: str, max_revisions: int = 2) -> dict:
    """Propose, gate, and on rejection revise up to max_revisions times."""
    prior_reason: str | None = None
    decision: dict = {}
    for attempt in range(max_revisions + 1):
        action = propose_action(goal, prior_rejection=prior_reason)
        decision = tiered_gate(action, attempt=attempt)
        decision["attempt"] = attempt
        log_decision(decision)
        if decision["decision"] == "approve":
            return decision
        prior_reason = decision["reason"]
    return {**decision, "final": "max_revisions_reached"}


def main() -> None:
    goals = [
        "Look up the weather in Seattle for the customer's trip planning.",
        "Schedule a reminder for the customer to check in 24 hours before their flight.",
        "Send a marketing email to the customer about premium upgrade options.",
    ]

    for goal in goals:
        print(f"\n=== Goal: {goal} ===")
        outcome = run_with_revision(goal, max_revisions=1)
        print(f"[final] {outcome['decision']} ({outcome['reason']})")

    print(f"\n=== Audit log ({GATE_LOG_PATH.name}) ===")
    for line in GATE_LOG_PATH.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        print(
            f"  [{record['risk_tier']:6s}] {record['decision']:8s} "
            f"attempt={record.get('attempt', '?')} "
            f"action={record['action'][:140]}"
        )


if __name__ == "__main__":
    main()
