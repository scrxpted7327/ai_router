"""
pi CLI provider — routes requests through `pi --model <model> <prompt>`.
Converts chat completions messages to a single prompt string, then invokes
the `pi` binary and wraps the output in OpenAI-format responses.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import AsyncIterator

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHFABCDJ]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _messages_to_prompt(messages: list[dict]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        if role == "system":
            parts.append(f"[System]\n{content}")
        elif role == "assistant":
            parts.append(f"[Assistant]\n{content}")
        else:
            parts.append(content)
    return "\n\n".join(parts)


async def chat(model_id: str, body: dict) -> dict:
    prompt = _messages_to_prompt(body.get("messages", []))
    proc = await asyncio.create_subprocess_exec(
        "pi", "--model", model_id, prompt,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=120)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise RuntimeError("pi CLI timed out after 120s")

    if proc.returncode != 0:
        msg = err.decode(errors="replace").strip() or "pi CLI exited non-zero"
        raise RuntimeError(msg)

    text = _strip_ansi(out.decode(errors="replace")).strip()
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def stream(model_id: str, body: dict) -> AsyncIterator[str]:
    prompt = _messages_to_prompt(body.get("messages", []))
    proc = await asyncio.create_subprocess_exec(
        "pi", "--model", model_id, prompt,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    assert proc.stdout is not None
    try:
        async for raw in proc.stdout:
            text = _strip_ansi(raw.decode(errors="replace"))
            if not text:
                continue
            chunk = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
    finally:
        try:
            await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            proc.kill()

    done_chunk = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done_chunk)}\n\n"
    yield "data: [DONE]\n\n"
