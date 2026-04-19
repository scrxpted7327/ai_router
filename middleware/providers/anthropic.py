"""
Anthropic provider — translates OpenAI chat/tool format to Anthropic SDK format
and streams back OpenAI-compatible SSE.
"""
from __future__ import annotations
import json
import uuid
from typing import Any, AsyncIterator

import anthropic as sdk


def _client(api_key: str) -> sdk.AsyncAnthropic:
    return sdk.AsyncAnthropic(api_key=api_key)


def _convert_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Split out system prompt; convert tool results to Anthropic format."""
    system: str | None = None
    converted = []
    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            system = m.get("content", "")
            continue
        if role == "tool":
            # OpenAI tool result → Anthropic tool_result
            converted.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id", ""),
                    "content": m.get("content", ""),
                }],
            })
            continue
        # Handle assistant messages with tool_calls
        content = m.get("content", "")
        tool_calls = m.get("tool_calls", [])
        if tool_calls:
            blocks: list[dict] = []
            if content:
                blocks.append({"type": "text", "text": content})
            for tc in tool_calls:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", f"tu_{uuid.uuid4().hex[:8]}"),
                    "name": fn.get("name", ""),
                    "input": args,
                })
            converted.append({"role": role, "content": blocks})
        else:
            converted.append({"role": role, "content": content})
    return system, converted


def _convert_tools(tools: list[dict]) -> list[dict]:
    """OpenAI tool schema → Anthropic tool schema."""
    result = []
    for t in tools:
        if t.get("type") != "function":
            continue
        fn = t["function"]
        result.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return result


async def chat(model_id: str, body: dict, api_key: str) -> dict:
    client = _client(api_key)
    system, messages = _convert_messages(body.get("messages", []))
    tools = _convert_tools(body.get("tools", []))

    kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": body.get("max_tokens") or 8192,
    }
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = tools

    resp = await client.messages.create(**kwargs)

    # Translate back to OpenAI response format
    content_blocks = resp.content
    text = ""
    tool_calls = []
    for block in content_blocks:
        if block.type == "text":
            text += block.text
        elif block.type == "tool_use":
            tool_calls.append({
                "id": block.id,
                "type": "function",
                "function": {"name": block.name, "arguments": json.dumps(block.input)},
            })

    message: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": resp.id,
        "object": "chat.completion",
        "model": model_id,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": resp.stop_reason}],
        "usage": {
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
            "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
        },
    }


async def stream(model_id: str, body: dict, api_key: str) -> AsyncIterator[str]:
    client = _client(api_key)
    system, messages = _convert_messages(body.get("messages", []))
    tools = _convert_tools(body.get("tools", []))
    stream_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"

    kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": body.get("max_tokens") or 8192,
    }
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = tools

    tool_call_map: dict[str, dict] = {}

    async with client.messages.stream(**kwargs) as s:
        async for event in s:
            etype = type(event).__name__

            if etype == "RawContentBlockStartEvent":
                block = event.content_block
                if block.type == "tool_use":
                    tool_call_map[str(event.index)] = {
                        "id": block.id, "name": block.name, "index": event.index
                    }
                    chunk = _tool_call_start_chunk(stream_id, model_id, event.index, block)
                    yield f"data: {json.dumps(chunk)}\n\n"

            elif etype == "RawContentBlockDeltaEvent":
                delta = event.delta
                if delta.type == "text_delta":
                    chunk = _text_delta_chunk(stream_id, model_id, delta.text)
                    yield f"data: {json.dumps(chunk)}\n\n"
                elif delta.type == "input_json_delta":
                    tc = tool_call_map.get(str(event.index))
                    if tc:
                        chunk = _tool_arg_delta_chunk(stream_id, model_id, tc["index"], delta.partial_json)
                        yield f"data: {json.dumps(chunk)}\n\n"

            elif etype == "RawMessageStopEvent":
                yield f"data: {json.dumps(_stop_chunk(stream_id, model_id))}\n\n"

    yield "data: [DONE]\n\n"


# ── SSE chunk helpers ─────────────────────────────────────────────────────────

def _base_chunk(stream_id: str, model: str) -> dict:
    return {"id": stream_id, "object": "chat.completion.chunk", "model": model, "choices": []}


def _text_delta_chunk(stream_id: str, model: str, text: str) -> dict:
    c = _base_chunk(stream_id, model)
    c["choices"] = [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
    return c


def _tool_call_start_chunk(stream_id: str, model: str, idx: int, block: Any) -> dict:
    c = _base_chunk(stream_id, model)
    c["choices"] = [{"index": 0, "delta": {"tool_calls": [{
        "index": idx, "id": block.id, "type": "function",
        "function": {"name": block.name, "arguments": ""},
    }]}, "finish_reason": None}]
    return c


def _tool_arg_delta_chunk(stream_id: str, model: str, idx: int, partial: str) -> dict:
    c = _base_chunk(stream_id, model)
    c["choices"] = [{"index": 0, "delta": {"tool_calls": [{
        "index": idx, "function": {"arguments": partial},
    }]}, "finish_reason": None}]
    return c


def _stop_chunk(stream_id: str, model: str) -> dict:
    c = _base_chunk(stream_id, model)
    c["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    return c
