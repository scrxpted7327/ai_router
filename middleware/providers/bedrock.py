"""Amazon Bedrock provider via the Converse APIs."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import queue
import uuid
from typing import Any, AsyncIterator

import boto3
from botocore.config import Config


def _session(options: dict[str, str] | None) -> boto3.session.Session:
    options = options or {}
    profile = (options.get("profile") or os.getenv("AWS_PROFILE") or "").strip() or None
    return boto3.session.Session(profile_name=profile)


def _client(options: dict[str, str] | None):
    options = options or {}
    region = options.get("region") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if not region:
        raise RuntimeError("AWS_REGION is required for Amazon Bedrock")

    kwargs: dict[str, Any] = {
        "region_name": region,
        "config": Config(retries={"max_attempts": 3, "mode": "standard"}),
    }
    endpoint = (options.get("endpoint") or "").strip()
    if endpoint:
        kwargs["endpoint_url"] = endpoint

    bearer_token = (options.get("bearer_token") or os.getenv("AWS_BEARER_TOKEN_BEDROCK") or "").strip()
    if bearer_token:
        kwargs["aws_access_key_id"] = "token"
        kwargs["aws_secret_access_key"] = "token"
        kwargs["aws_session_token"] = bearer_token

    return _session(options).client("bedrock-runtime", **kwargs)


def _decode_image_data(url: str) -> dict | None:
    if not url.startswith("data:") or ";base64," not in url:
        return None
    header, payload = url.split(",", 1)
    media_type = header[5:].split(";", 1)[0]
    image_format = media_type.split("/", 1)[-1].lower()
    if image_format == "jpeg":
        image_format = "jpg"
    try:
        data = base64.b64decode(payload)
    except Exception:
        return None
    return {"format": image_format or "png", "source": {"bytes": data}}


def _content_blocks(content: Any) -> list[dict]:
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list):
        return [{"text": str(content)}]

    blocks: list[dict] = []
    for part in content:
        if not isinstance(part, dict):
            blocks.append({"text": str(part)})
            continue
        part_type = part.get("type")
        if part_type == "text":
            blocks.append({"text": str(part.get("text") or "")})
            continue
        if part_type == "image_url":
            image_url = part.get("image_url") or {}
            url = str(image_url.get("url") or "")
            decoded = _decode_image_data(url)
            if decoded:
                blocks.append({"image": decoded})
            continue
    return blocks or [{"text": ""}]


def _convert_messages(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    system: list[dict] = []
    converted: list[dict] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = _content_blocks(message.get("content"))
        if role == "system":
            system.extend(content)
            continue
        if role == "tool":
            role = "user"
        if role not in {"user", "assistant"}:
            role = "user"
        converted.append({"role": role, "content": content})
    return system, converted


def _inference_config(body: dict) -> dict:
    config: dict[str, Any] = {"maxTokens": int(body.get("max_tokens") or body.get("max_completion_tokens") or 8192)}
    if body.get("temperature") is not None:
        config["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        config["topP"] = body["top_p"]
    if body.get("stop"):
        stop = body["stop"]
        config["stopSequences"] = stop if isinstance(stop, list) else [stop]
    return config


def _base_chunk(stream_id: str, model_id: str) -> dict:
    return {"id": stream_id, "object": "chat.completion.chunk", "model": model_id, "choices": []}


def _text_chunk(stream_id: str, model_id: str, text: str) -> dict:
    chunk = _base_chunk(stream_id, model_id)
    chunk["choices"] = [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
    return chunk


def _stop_chunk(stream_id: str, model_id: str, reason: str = "stop") -> dict:
    chunk = _base_chunk(stream_id, model_id)
    chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": reason}]
    return chunk


def _message_payload(model_id: str, body: dict) -> dict:
    system, messages = _convert_messages(body.get("messages", []))
    payload: dict[str, Any] = {
        "modelId": model_id,
        "messages": messages,
        "inferenceConfig": _inference_config(body),
    }
    if system:
        payload["system"] = system
    return payload


def _message_text(content: list[dict]) -> str:
    parts: list[str] = []
    for block in content or []:
        text = block.get("text")
        if text:
            parts.append(text)
    return "".join(parts)


async def chat(model_id: str, body: dict, options: dict[str, str] | None = None) -> dict:
    client = _client(options)
    payload = _message_payload(model_id, body)
    response = await asyncio.to_thread(client.converse, **payload)

    output = response.get("output") or {}
    message = output.get("message") or {}
    content = message.get("content") or []
    text = _message_text(content)
    usage = response.get("usage") or {}
    stop_reason = str(response.get("stopReason") or "end_turn").lower()
    finish_reason = "stop" if stop_reason in {"stop", "end_turn"} else stop_reason

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "model": model_id,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": int(usage.get("inputTokens") or 0),
            "completion_tokens": int(usage.get("outputTokens") or 0),
            "total_tokens": int(usage.get("totalTokens") or 0),
        },
    }


async def stream(model_id: str, body: dict, options: dict[str, str] | None = None) -> AsyncIterator[str]:
    client = _client(options)
    payload = _message_payload(model_id, body)
    stream_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    event_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

    def _worker() -> None:
        try:
            response = client.converse_stream(**payload)
            for event in response.get("stream") or []:
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta") or {}
                    text = delta.get("text")
                    if text:
                        event_queue.put(("text", text))
                elif "messageStop" in event:
                    stop_reason = str(event["messageStop"].get("stopReason") or "stop").lower()
                    event_queue.put(("stop", stop_reason))
            event_queue.put(("done", None))
        except Exception as exc:
            event_queue.put(("error", exc))

    task = asyncio.create_task(asyncio.to_thread(_worker))
    seen_stop = False
    try:
        while True:
            kind, payload_item = await asyncio.to_thread(event_queue.get)
            if kind == "text":
                yield f"data: {json.dumps(_text_chunk(stream_id, model_id, payload_item))}\n\n"
                continue
            if kind == "stop":
                seen_stop = True
                finish_reason = "stop" if payload_item in {"stop", "end_turn"} else str(payload_item)
                yield f"data: {json.dumps(_stop_chunk(stream_id, model_id, finish_reason))}\n\n"
                continue
            if kind == "error":
                raise payload_item
            if kind == "done":
                if not seen_stop:
                    yield f"data: {json.dumps(_stop_chunk(stream_id, model_id))}\n\n"
                break
    finally:
        await asyncio.gather(task, return_exceptions=True)
    yield "data: [DONE]\n\n"
