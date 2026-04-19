"""
Anthropic Messages API — full gateway proxy.

Accepts requests in native Anthropic SDK format (x-api-key = gateway bearer token),
routes through the model registry (any provider — Gemini, GPT, Groq, Copilot, etc.),
and returns responses in Anthropic Messages API format so Claude Code works with
every model the gateway supports.

Claude Code / claude CLI config:
  ANTHROPIC_BASE_URL=https://ai.scrxpted.cc/anthropic
  ANTHROPIC_API_KEY=air_...   (your gateway token)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select

from . import registry as reg
from .db import GatewayApiToken, SessionLocal, User
from .providers import gemini as gemini_provider
from .providers import openai_compat

log = logging.getLogger("ai_router")

router = APIRouter(prefix="/anthropic")


# ── Auth ──────────────────────────────────────────────────────────────────────

async def _auth(request: Request) -> User:
    raw = request.headers.get("x-api-key", "").strip()
    if not raw:
        raise HTTPException(status_code=401, detail="Missing x-api-key header")
    digest = hashlib.sha256(raw.encode()).hexdigest()
    async with SessionLocal() as db:
        tok = (await db.execute(
            select(GatewayApiToken).where(GatewayApiToken.token_digest == digest)
        )).scalars().first()
        if not tok:
            raise HTTPException(status_code=401, detail="Invalid API token")
        user = await db.get(User, tok.user_id)
        if not user or not user.is_whitelisted:
            raise HTTPException(status_code=403, detail="Account not whitelisted")
    return user


# ── Anthropic request → internal OpenAI-compat body ──────────────────────────

def _to_openai_body(body: dict) -> dict:
    messages = []

    system = body.get("system")
    if isinstance(system, str) and system:
        messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        text = "\n".join(b.get("text", "") for b in system if b.get("type") == "text")
        if text:
            messages.append({"role": "system", "content": text})

    for m in body.get("messages", []):
        role = m["role"]
        content = m["content"]
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            parts: list = []
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    parts.append({"type": "text", "text": block["text"]})
                elif btype == "image":
                    src = block.get("source", {})
                    if src.get("type") == "base64":
                        parts.append({"type": "image_url", "image_url": {
                            "url": f"data:{src['media_type']};base64,{src['data']}"
                        }})
                elif btype == "tool_use":
                    # assistant tool call
                    parts.append({"type": "text", "text": f"[tool_use:{block.get('name')}]"})
                elif btype == "tool_result":
                    parts.append({"type": "text", "text": str(block.get("content", ""))})
            messages.append({
                "role": role,
                "content": parts if len(parts) != 1 or parts[0]["type"] != "text"
                           else parts[0]["text"],
            })

    oai: dict = {
        "model": body.get("model", "claude-sonnet-4-5"),
        "messages": messages,
        "max_tokens": body.get("max_tokens", 8192),
        "stream": body.get("stream", False),
    }
    if body.get("temperature") is not None:
        oai["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        oai["top_p"] = body["top_p"]
    return oai


# ── OpenAI response → Anthropic Messages API response ────────────────────────

def _to_anthropic_response(oai: dict, model: str, msg_id: str) -> dict:
    choice = oai["choices"][0]
    text = (choice.get("message") or {}).get("content") or ""
    finish = choice.get("finish_reason", "stop")
    usage = oai.get("usage", {})
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model,
        "stop_reason": "end_turn" if finish == "stop" else finish,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ── OpenAI SSE stream → Anthropic SSE stream ─────────────────────────────────

async def _as_anthropic_stream(
    oai_stream: AsyncIterator[str], model: str, msg_id: str
) -> AsyncIterator[str]:
    yield f"event: message_start\ndata: {json.dumps({'type':'message_start','message':{'id':msg_id,'type':'message','role':'assistant','content':[],'model':model,'stop_reason':None,'stop_sequence':None,'usage':{'input_tokens':0,'output_tokens':0}}})}\n\n"
    yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}})}\n\n"
    yield f"event: ping\ndata: {json.dumps({'type':'ping'})}\n\n"

    async for sse_line in oai_stream:
        # providers yield full SSE lines: "data: {...}\n\n"
        for line in sse_line.splitlines():
            if not line.startswith("data: "):
                continue
            raw = line[6:]
            if raw.strip() == "[DONE]":
                continue
            try:
                chunk = json.loads(raw)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                text = delta.get("content")
                if text:
                    yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':text}})}\n\n"
            except (json.JSONDecodeError, IndexError):
                pass

    yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':0})}\n\n"
    yield f"event: message_delta\ndata: {json.dumps({'type':'message_delta','delta':{'stop_reason':'end_turn','stop_sequence':None},'usage':{'output_tokens':0}})}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type':'message_stop'})}\n\n"


# ── Provider dispatch (mirrors app.py logic) ──────────────────────────────────

async def _complete(entry: reg.ModelEntry, body: dict) -> dict:
    if entry.provider == "gemini":
        return await gemini_provider.chat(entry.model_id, body, entry.api_key)
    return await openai_compat.chat(
        entry.model_id, body, entry.api_key, entry.base_url, entry.extra_headers
    )


def _stream(entry: reg.ModelEntry, body: dict) -> AsyncIterator[str]:
    if entry.provider == "gemini":
        return gemini_provider.stream(entry.model_id, body, entry.api_key)
    return openai_compat.stream(
        entry.model_id, body, entry.api_key, entry.base_url, entry.extra_headers
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/v1/messages")
async def messages(request: Request):
    await _auth(request)

    body = await request.json()
    requested_model = body.get("model", "")
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    is_stream = body.get("stream", False)

    entry = reg.get(requested_model)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Model '{requested_model}' not found in gateway registry")

    if not entry.api_key or "REPLACE_ME" in entry.api_key:
        raise HTTPException(status_code=503, detail=f"No API key configured for '{requested_model}'")

    oai_body = _to_openai_body(body)
    oai_body["model"] = entry.model_id
    oai_body["stream"] = is_stream

    log.info("Anthropic proxy: '%s' → %s (%s)", requested_model, entry.model_id, entry.provider)

    if is_stream:
        oai_iter = _stream(entry, oai_body)
        ant_iter = _as_anthropic_stream(oai_iter, requested_model, msg_id)
        return StreamingResponse(
            ant_iter,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    oai_resp = await _complete(entry, oai_body)
    return Response(
        content=json.dumps(_to_anthropic_response(oai_resp, requested_model, msg_id)),
        media_type="application/json",
    )


@router.get("/v1/models")
async def models(request: Request):
    await _auth(request)
    return {
        "object": "list",
        "data": [
            {
                "id": m["id"],
                "object": "model",
                "display_name": m["name"],
                "owned_by": m["owned_by"],
                "reasoning": m["reasoning"],
                "vision": m["vision"],
                "context_window": m["context_window"],
                "max_tokens": m["max_tokens"],
            }
            for m in reg.list_models()
        ],
    }
