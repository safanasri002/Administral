"""Mistral served locally (Ollama) — the shared fallback when the cloud API is down.

One small client for every module that can degrade to the local model instead
of failing: the citizen assistant's generation, today. It answers in the same
shape the cloud call does (plain text, or JSON matching a schema), so callers —
and the frontend behind them — cannot tell which model replied.

Reuses the locality guard of `ai.inconsistency.local_agent`: a misconfigured
URL must never turn "local fallback" into a call to some public endpoint.
"""

from __future__ import annotations

import httpx

from app.core.config import settings
from app.modules.ai.inconsistency.local_agent import _assert_local


def enabled() -> bool:
    return settings.local_llm_enabled


def chat(
    messages: list[dict],
    *,
    json_schema: dict | None = None,
    temperature: float = 0.1,
    num_predict: int = 700,
    num_ctx: int = 8192,
) -> str:
    """One chat completion from the local model. Raises on any failure.

    `json_schema` makes Ollama constrain the output: a 7B model in free JSON
    mode drifts from the contract often enough that parsing would fail.
    """
    _assert_local(settings.local_llm_url)
    body: dict = {
        "model": settings.local_llm_model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict, "num_ctx": num_ctx},
        "keep_alive": "30m",
    }
    if json_schema is not None:
        body["format"] = json_schema
    response = httpx.post(settings.local_llm_url, json=body, timeout=settings.local_llm_timeout_seconds)
    response.raise_for_status()
    return response.json()["message"]["content"]
