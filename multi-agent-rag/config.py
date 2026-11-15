
"""
Single place for the Groq client and model choices.

All agents share one Groq client.
Models can be changed through environment variables.
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

# Load .env from this project's directory
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# API key
_api_key = os.getenv("GROQ_API_KEY")

if not _api_key:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Create a .env file with "
        "GROQ_API_KEY=gsk_... (see .env.example)."
    )

# Shared Groq client
client = Groq(
    api_key=_api_key,
    max_retries=3,
    timeout=60.0
)

# Free-tier compatible model configuration
# Both models use the same free model initially.
MODEL_SMART = os.getenv(
    "GROQ_MODEL_SMART",
    "openai/gpt-oss-20b"
)

MODEL_FAST = os.getenv(
    "GROQ_MODEL_FAST",
    "openai/gpt-oss-20b"
)


def parse_json_array(text: str) -> list:
    """
    Pull a JSON array out of an LLM reply,
    tolerating markdown fences and surrounding prose.
    """
    text = text.replace("```json", "").replace("```", "").strip()

    start, end = text.find("["), text.rfind("]")

    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON array found in model output")

    return json.loads(text[start:end + 1])


def parse_json_object(text: str) -> dict:
    """Extract a JSON object from an LLM reply."""

    text = text.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError("no JSON object found in model output")

    return json.loads(match.group(0))

print("MODELS:", MODEL_SMART, MODEL_FAST)
