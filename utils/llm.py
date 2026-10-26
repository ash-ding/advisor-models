"""Route black-box LLM calls to Claude on Vertex AI.

Every student / judge / summarizer call in this repo goes through
`chat_completion`, which maps OpenAI model names onto Claude models served by
Vertex AI and calls them via litellm. Calls that carry an `api_base` pointing at
a local server (the vLLM-served advisor) are passed through untouched.

Environment variables:
- `VERTEX_PROJECT` / `ANTHROPIC_VERTEX_PROJECT_ID`: GCP project (required).
- `VERTEX_LOCATION` / `CLOUD_ML_REGION`: Vertex region (default `global`).
- `GOOGLE_APPLICATION_CREDENTIALS`: service-account key file.
- `CLAUDE_SMALL_MODEL`: replaces mini/nano OpenAI models (default `claude-haiku-4-5`).
- `CLAUDE_LARGE_MODEL`: replaces full-size OpenAI models (default `claude-sonnet-4-6`).
- `CLAUDE_MAX_TOKENS`: default `max_tokens` for Claude calls (default 8192).
"""

import os
import re
from typing import Any, Dict, List, Optional

import litellm

VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT") or os.environ.get(
    "ANTHROPIC_VERTEX_PROJECT_ID"
)
VERTEX_LOCATION = (
    os.environ.get("VERTEX_LOCATION") or os.environ.get("CLOUD_ML_REGION") or "global"
)
CLAUDE_SMALL_MODEL = os.environ.get("CLAUDE_SMALL_MODEL", "claude-haiku-4-5")
CLAUDE_LARGE_MODEL = os.environ.get("CLAUDE_LARGE_MODEL", "claude-sonnet-4-6")
CLAUDE_MAX_TOKENS = int(os.environ.get("CLAUDE_MAX_TOKENS", "8192"))

_OPENAI_NAME = re.compile(r"^(openai/)?(gpt-|o\d)")
_SMALL_TIER = re.compile(r"(mini|nano)")


def resolve_model(model: str) -> str:
    """Map a model name from the original code onto a litellm Vertex Claude id.

    - `vertex_ai/...` is returned unchanged.
    - OpenAI names (`gpt-*`, `o1`/`o3`/`o4-*`, `openai/*`) map to
      `CLAUDE_SMALL_MODEL` for mini/nano variants and `CLAUDE_LARGE_MODEL` otherwise.
    - `claude-*` / `anthropic/claude-*` keep the Claude model but go through Vertex.
    - Anything else (e.g. `hosted_vllm/...`, `gemini/...`) is returned unchanged.
    """
    if model.startswith("vertex_ai/"):
        return model
    if _OPENAI_NAME.match(model):
        tier = CLAUDE_SMALL_MODEL if _SMALL_TIER.search(model) else CLAUDE_LARGE_MODEL
        return f"vertex_ai/{tier}"
    name = model.split("/", 1)[1] if model.startswith("anthropic/") else model
    if name.startswith("claude-"):
        return f"vertex_ai/{name}"
    return model


def is_vertex_claude(model: str) -> bool:
    return resolve_model(model).startswith("vertex_ai/claude")


def vertex_kwargs(model: str) -> Dict[str, Any]:
    """Extra litellm kwargs needed to reach `model` on Vertex ({} if not Claude).

    For callers that hand a model name to third-party code built on litellm
    (e.g. mini-swe-agent's `LitellmModel(model_kwargs=...)`).
    """
    if not is_vertex_claude(model):
        return {}
    return {
        "vertex_project": VERTEX_PROJECT,
        "vertex_location": VERTEX_LOCATION,
        "vertex_credentials": os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
    }


def chat_completion(
    model: str,
    messages: List[Dict[str, Any]],
    api_base: Optional[str] = None,
    **kwargs: Any,
):
    """Drop-in replacement for `litellm.completion` / OpenAI chat completions.

    Returns the litellm response object, which has the same
    `response.choices[0].message.content` shape as the OpenAI client.
    """
    resolved = resolve_model(model)
    if not resolved.startswith("vertex_ai/claude"):
        # Local advisor endpoints and other non-OpenAI providers are untouched.
        if api_base is not None:
            kwargs["api_base"] = api_base
        return litellm.completion(model=resolved, messages=messages, **kwargs)

    # OpenAI-only parameters Claude does not accept.
    for key in ("base_url", "reasoning_effort", "max_completion_tokens", "seed"):
        kwargs.pop(key, None)
    if kwargs.get("temperature") is not None:
        kwargs["temperature"] = min(float(kwargs["temperature"]), 1.0)
    kwargs.setdefault("max_tokens", CLAUDE_MAX_TOKENS)
    return litellm.completion(
        model=resolved, messages=messages, **vertex_kwargs(model), **kwargs
    )


class _Completions:
    def create(self, model: str, messages: List[Dict[str, Any]], **kwargs: Any):
        return chat_completion(model, messages, **kwargs)


class _Chat:
    def __init__(self):
        self.completions = _Completions()


class ClaudeClient:
    """Stand-in for `openai.OpenAI()`: `client.chat.completions.create(...)`
    is served by Claude on Vertex AI via `chat_completion`."""

    def __init__(self, *args: Any, **kwargs: Any):
        self.chat = _Chat()


def completion_cost(response) -> float:
    """litellm cost lookup that tolerates models missing from its price table."""
    try:
        return litellm.completion_cost(completion_response=response) or 0.0
    except Exception:
        return 0.0
