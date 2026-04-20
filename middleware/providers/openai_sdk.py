"""
Direct OpenAI-compatible SDK provider for first-party OpenAI endpoints.

Differs from openai_compat in that it:
  - Strips params that relay providers receive but can't forward
  - Properly handles reasoning_effort for o-series models
  - Never passes `thinking` (Anthropic-style) — strips it silently

Used for: openai, xai, deepseek, cohere, mistral, together, fireworks, perplexity, zai
NOT used for: openrouter, kilo, opencode (those stay on openai_compat for pass-through semantics)
"""
from __future__ import annotations
import json
from typing import Any, AsyncIterator

from openai import AsyncOpenAI, APIStatusError


def _client(api_key: str, base_url: str | None, extra_headers: dict | None) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=extra_headers or {},
    )


async def chat(
    model_id: str,
    body: dict,
    api_key: str,
    base_url: str | None,
    extra_headers: dict | None,
    supports_effort: bool = True,
) -> dict:
    client = _client(api_key, base_url, extra_headers)
    params = _build_params(model_id, body, stream=False, supports_effort=supports_effort)
    resp = await client.chat.completions.create(**params)
    return resp.model_dump()


async def stream(
    model_id: str,
    body: dict,
    api_key: str,
    base_url: str | None,
    extra_headers: dict | None,
    supports_effort: bool = True,
) -> AsyncIterator[str]:
    client = _client(api_key, base_url, extra_headers)
    params = _build_params(model_id, body, stream=True, supports_effort=supports_effort)
    async with client.chat.completions.stream(**params) as s:
        async for event in s:
            chunk = event.model_dump()
            yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


def _build_params(model_id: str, body: dict, stream: bool, supports_effort: bool) -> dict:
    allowed = {
        "messages", "temperature", "top_p", "max_tokens", "max_completion_tokens",
        "tools", "tool_choice", "response_format", "stop", "n", "presence_penalty",
        "frequency_penalty", "logit_bias", "user", "seed",
    }
    params: dict[str, Any] = {k: v for k, v in body.items() if k in allowed}
    params["model"] = model_id
    params["stream"] = stream

    if supports_effort:
        effort = body.get("reasoning_effort")
        if effort and effort not in ("", "default"):
            params["reasoning_effort"] = effort

    return params
