"""
Google Gemini provider — translates OpenAI format ↔ Google GenAI SDK.
"""
from __future__ import annotations
import json
import uuid
from typing import Any, AsyncIterator

from google import genai
from google.genai import types


def _client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def _convert_messages(messages: list[dict]) -> tuple[str | None, list[types.Content]]:
    system: str | None = None
    contents: list[types.Content] = []
    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            system = m.get("content", "")
            continue
        gemini_role = "model" if role == "assistant" else "user"
        content_val = m.get("content", "")
        if role == "tool":
            contents.append(types.Content(role="user", parts=[
                types.Part(function_response=types.FunctionResponse(
                    id=m.get("tool_call_id", ""),
                    name="tool_result",
                    response={"result": m.get("content", "")},
                ))
            ]))
            continue
        tool_calls = m.get("tool_calls", [])
        if tool_calls:
            parts = []
            if content_val:
                parts.append(types.Part(text=content_val))
            for tc in tool_calls:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                parts.append(types.Part(function_call=types.FunctionCall(
                    id=tc.get("id", ""),
                    name=fn.get("name", ""),
                    args=args,
                )))
            contents.append(types.Content(role=gemini_role, parts=parts))
        else:
            contents.append(types.Content(role=gemini_role, parts=[types.Part(text=content_val)]))
    return system, contents


def _convert_tools(tools: list[dict]) -> list[types.Tool] | None:
    if not tools:
        return None
    declarations = []
    for t in tools:
        if t.get("type") != "function":
            continue
        fn = t["function"]
        declarations.append(types.FunctionDeclaration(
            name=fn["name"],
            description=fn.get("description", ""),
            parameters=fn.get("parameters"),
        ))
    return [types.Tool(function_declarations=declarations)] if declarations else None


async def chat(model_id: str, body: dict, api_key: str) -> dict:
    client = _client(api_key)
    system, contents = _convert_messages(body.get("messages", []))
    tools = _convert_tools(body.get("tools", []))

    config_kwargs: dict[str, Any] = {
        "max_output_tokens": body.get("max_tokens") or 8192,
    }
    if system:
        config_kwargs["system_instruction"] = system
    if tools:
        config_kwargs["tools"] = tools

    resp = await client.aio.models.generate_content(
        model=model_id,
        contents=contents,
        config=types.GenerateContentConfig(**config_kwargs),
    )

    text = ""
    tool_calls = []
    for part in (resp.candidates[0].content.parts if resp.candidates else []):
        if part.text:
            text += part.text
        if part.function_call:
            fc = part.function_call
            tool_calls.append({
                "id": fc.id or f"fc_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": fc.name, "arguments": json.dumps(dict(fc.args or {}))},
            })

    message: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = resp.usage_metadata
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "model": model_id,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": usage.prompt_token_count if usage else 0,
            "completion_tokens": usage.candidates_token_count if usage else 0,
            "total_tokens": usage.total_token_count if usage else 0,
        },
    }


async def stream(model_id: str, body: dict, api_key: str) -> AsyncIterator[str]:
    client = _client(api_key)
    system, contents = _convert_messages(body.get("messages", []))
    tools = _convert_tools(body.get("tools", []))
    stream_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"

    config_kwargs: dict[str, Any] = {
        "max_output_tokens": body.get("max_tokens") or 8192,
    }
    if system:
        config_kwargs["system_instruction"] = system
    if tools:
        config_kwargs["tools"] = tools

    async for chunk in await client.aio.models.generate_content_stream(
        model=model_id,
        contents=contents,
        config=types.GenerateContentConfig(**config_kwargs),
    ):
        for part in (chunk.candidates[0].content.parts if chunk.candidates else []):
            if part.text:
                sse = {
                    "id": stream_id, "object": "chat.completion.chunk", "model": model_id,
                    "choices": [{"index": 0, "delta": {"content": part.text}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(sse)}\n\n"
            if part.function_call:
                fc = part.function_call
                sse = {
                    "id": stream_id, "object": "chat.completion.chunk", "model": model_id,
                    "choices": [{"index": 0, "delta": {"tool_calls": [{
                        "index": 0,
                        "id": fc.id or f"fc_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {"name": fc.name, "arguments": json.dumps(dict(fc.args or {}))},
                    }]}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(sse)}\n\n"

    stop = {
        "id": stream_id, "object": "chat.completion.chunk", "model": model_id,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(stop)}\n\n"
    yield "data: [DONE]\n\n"
