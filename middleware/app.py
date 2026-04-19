"""
Unified AI Gateway — OpenAI-compatible FastAPI server.

Supports:
  POST /v1/chat/completions    — standard chat (all harnesses)
  POST /v1/responses           — Cursor agent mode (Responses API)
  GET  /v1/models              — model listing
  GET  /health

Dashboard:  GET /  — OpenClaw-style control UI (health, models, memory, connect)

Configure Cursor:  Settings → Models → Add Model
  Base URL:  http://localhost:4000/v1
  API Key:   gateway Bearer token from `manage_users.py token-create`
  Model:     claude, gpt-4o, gemini, groq, free, copilot, ...
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import registry as reg
from .auth import get_db, require_whitelisted, router as auth_router
from .compactor import compact, needs_compaction
from .db import Conversation, SessionLocal, init_db
from .format_adapter import normalise_request, stream_as_responses_api
from .router import route
from .providers import anthropic as anthropic_provider
from .providers import gemini as gemini_provider
from .providers import openai_compat

ENV_PATH = Path(__file__).parent.parent / ".env"
STATIC_DASHBOARD = Path(__file__).parent / "static" / "dashboard"

log = logging.getLogger("ai_router")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "*").strip()
    if raw == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()] or ["*"]


app = FastAPI(title="Unified AI Gateway", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth_router)


@app.on_event("startup")
async def startup() -> None:
    await init_db()
    reg.init(str(ENV_PATH))
    log.info("Registry loaded — %d models available", len(reg.list_models()))


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "models": len(reg.list_models())}


@app.get("/")
async def dashboard() -> FileResponse:
    """OpenClaw-style control UI (static SPA)."""
    return FileResponse(STATIC_DASHBOARD / "index.html")


app.mount(
    "/static/dashboard",
    StaticFiles(directory=str(STATIC_DASHBOARD)),
    name="dashboard_static",
)


# ── Model listing ─────────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models() -> dict:
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "owned_by": "ai-router"}
            for m in reg.list_models()
        ],
    }


# ── Cursor Responses API endpoint ─────────────────────────────────────────────

@app.post("/v1/responses")
async def responses_endpoint(request: Request, user=Depends(require_whitelisted)) -> Response:
    body = await request.json()
    return await _handle(body, is_responses_api=True, user=user)


# ── Standard Chat Completions ─────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, user=Depends(require_whitelisted)) -> Response:
    body = await request.json()
    body, is_responses_api = normalise_request(body)
    return await _handle(body, is_responses_api=is_responses_api, user=user)


# ── Core dispatch ─────────────────────────────────────────────────────────────

async def _handle(body: dict, is_responses_api: bool, user=None) -> Response:
    messages = body.get("messages", [])
    do_stream = body.get("stream", True)

    # 1. Compact long histories via Groq
    if needs_compaction(messages):
        log.info("Compacting %d messages...", len(messages))
        messages = await compact(messages)
        body["messages"] = messages

    # 2. Resolve model — explicit name or auto-route
    requested = body.get("model", "")
    entry = reg.get(requested)
    if not entry:
        decision = route(messages)
        entry = reg.get(decision.primary)
        if not entry:
            # walk fallbacks
            for fb in decision.fallbacks:
                entry = reg.get(fb)
                if entry:
                    break
        if not entry:
            raise HTTPException(status_code=503, detail=f"No provider available for '{requested}'")
        log.info("Auto-routed '%s' → %s (%s)", requested, entry.model_id, entry.provider)
    else:
        log.info("Model '%s' → %s (%s)", requested, entry.model_id, entry.provider)

    if not entry.api_key or "REPLACE_ME" in entry.api_key:
        raise HTTPException(status_code=401,
                            detail=f"No API key for '{entry.provider}'. Run `python pi_auth.py`.")

    # 3. Dispatch to provider
    if do_stream:
        raw_stream = _stream(entry, body)
        if is_responses_api:
            raw_stream = stream_as_responses_api(raw_stream, entry.model_id)
        return StreamingResponse(raw_stream, media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})
    else:
        result = await _complete(entry, body)
        return Response(content=json.dumps(result), media_type="application/json")


async def _complete(entry: reg.ModelEntry, body: dict) -> dict:
    if entry.provider == "anthropic":
        return await anthropic_provider.chat(entry.model_id, body, entry.api_key)
    if entry.provider == "gemini":
        return await gemini_provider.chat(entry.model_id, body, entry.api_key)
    return await openai_compat.chat(
        entry.model_id, body, entry.api_key, entry.base_url, entry.extra_headers
    )


async def _stream(entry: reg.ModelEntry, body: dict) -> AsyncIterator[str]:
    if entry.provider == "anthropic":
        async for chunk in anthropic_provider.stream(entry.model_id, body, entry.api_key):
            yield chunk
    elif entry.provider == "gemini":
        async for chunk in gemini_provider.stream(entry.model_id, body, entry.api_key):
            yield chunk
    else:
        async for chunk in openai_compat.stream(
            entry.model_id, body, entry.api_key, entry.base_url, entry.extra_headers
        ):
            yield chunk
