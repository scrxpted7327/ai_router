"""
Translates between Cursor's Responses API format and standard Chat Completions.

Cursor agent mode POSTs to /v1/responses with:
  { "model": "...", "input": [...], "tools": [...], "stream": true }

Everything else uses /v1/chat/completions with:
  { "model": "...", "messages": [...], "tools": [...], "stream": true }

This module normalises all incoming requests to Chat Completions format,
and all outgoing SSE streams back to whatever format the caller expects.
"""
from __future__ import annotations
import json
import time
import uuid
from typing import Any, AsyncIterator


# ── Inbound: normalise to Chat Completions ────────────────────────────────────

def normalise_request(body: dict) -> tuple[dict, bool]:
    """
    Returns (chat_completions_body, is_responses_api).
    Mutates nothing; always returns a new dict.
    """
    if "input" in body and "messages" not in body:
        # Responses API → Chat Completions
        messages = _responses_input_to_messages(body["input"])
        normalised = {
            **{k: v for k, v in body.items() if k not in ("input",)},
            "messages": messages,
        }
        return normalised, True
    return body, False


def _responses_input_to_messages(input_items: list[dict]) -> list[dict]:
    messages = []
    for item in input_items:
        itype = item.get("type", "message")
        if itype == "message":
            role = item.get("role", "user")
            content = item.get("content", "")
            if isinstance(content, list):
                # multipart — flatten to text for now
                content = " ".join(
                    c.get("text", "") for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                )
            messages.append({"role": role, "content": content})
        elif itype == "function_call_output":
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": json.dumps(item.get("output", "")),
            })
    return messages


# ── Outbound: translate SSE streams ──────────────────────────────────────────

async def stream_as_responses_api(
    source: AsyncIterator[str],
    model: str,
) -> AsyncIterator[str]:
    """
    Wraps a Chat Completions SSE stream into Responses API SSE events.
    Cursor requires this format when it called /v1/responses.
    """
    response_id = f"resp_{uuid.uuid4().hex[:16]}"
    item_id = f"item_{uuid.uuid4().hex[:12]}"

    yield _sse({"type": "response.created", "response": {
        "id": response_id, "object": "realtime.response",
        "model": model, "status": "in_progress",
        "output": [], "created_at": int(time.time()),
    }})
    yield _sse({"type": "response.output_item.added", "response_id": response_id,
                "output_index": 0,
                "item": {"id": item_id, "type": "message",
                         "role": "assistant", "content": []}})
    yield _sse({"type": "response.content_part.added", "response_id": response_id,
                "item_id": item_id, "output_index": 0, "content_index": 0,
                "part": {"type": "output_text", "text": ""}})

    full_text = ""
    tool_calls: dict[int, dict] = {}

    async for line in source:
        if not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})

            # Text delta
            text = delta.get("content") or ""
            if text:
                full_text += text
                yield _sse({"type": "response.output_text.delta",
                            "response_id": response_id, "item_id": item_id,
                            "output_index": 0, "content_index": 0, "delta": text})

            # Tool call deltas
            for tc in delta.get("tool_calls", []):
                idx = tc.get("index", 0)
                if idx not in tool_calls:
                    tool_calls[idx] = {"id": tc.get("id", ""), "name": "", "arguments": ""}
                if tc.get("function", {}).get("name"):
                    tool_calls[idx]["name"] += tc["function"]["name"]
                if tc.get("function", {}).get("arguments"):
                    tool_calls[idx]["arguments"] += tc["function"]["arguments"]

    # Emit tool calls as function_call output items
    for idx, tc in tool_calls.items():
        yield _sse({"type": "response.output_item.added",
                    "response_id": response_id, "output_index": idx + 1,
                    "item": {"id": f"fc_{tc['id']}", "type": "function_call",
                             "call_id": tc["id"], "name": tc["name"],
                             "arguments": tc["arguments"], "status": "completed"}})

    yield _sse({"type": "response.content_part.done",
                "response_id": response_id, "item_id": item_id,
                "output_index": 0, "content_index": 0,
                "part": {"type": "output_text", "text": full_text}})
    yield _sse({"type": "response.output_item.done",
                "response_id": response_id, "output_index": 0,
                "item": {"id": item_id, "type": "message", "role": "assistant",
                         "status": "completed",
                         "content": [{"type": "output_text", "text": full_text}]}})
    yield _sse({"type": "response.completed",
                "response": {"id": response_id, "status": "completed",
                             "model": model, "output": [
                                 {"type": "message", "role": "assistant",
                                  "content": [{"type": "output_text", "text": full_text}]}
                             ]}})
    yield "data: [DONE]\n\n"


def _sse(obj: Any) -> str:
    return f"data: {json.dumps(obj)}\n\n"
