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

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
from sqlalchemy import select

import pi_auth
from . import registry as reg
from .anthropic_proxy import router as anthropic_router
from .auth import COOKIE_NAME, require_admin, require_whitelisted, router as auth_router
from .compactor import compact, needs_compaction
from .db import Session, SessionLocal, User, init_db
from .format_adapter import normalise_request, stream_as_responses_api
from .router import route
from .providers import gemini as gemini_provider
from .providers import openai_compat

ENV_PATH = Path(__file__).parent.parent / ".env"
STATIC_DASHBOARD = Path(__file__).parent / "static" / "dashboard"
STATIC_TERMINAL = Path(__file__).parent / "static" / "terminal"
PI_AUTH_SCRIPT = Path(__file__).parent.parent / "pi_auth.py"
PI_ALLOWED_COMMANDS = {"login", "/login"}

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
app.include_router(anthropic_router)


@app.on_event("startup")
async def startup() -> None:
    await init_db()
    reg.init(str(ENV_PATH))
    log.info("Registry loaded — %d models available", len(reg.list_models()))
    await _run_startup_pi_auth_check()


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

app.mount(
    "/static/terminal",
    StaticFiles(directory=str(STATIC_TERMINAL)),
    name="terminal_static",
)


def _env_target_path() -> Path:
    raw = os.getenv("PI_AUTH_ENV_FILE", "").strip()
    return Path(raw).expanduser().resolve() if raw else ENV_PATH


def _token_status_payload(providers: dict[str, dict], auth_path: Path) -> dict:
    rows = []
    for provider_key, mapping in pi_auth._PROVIDER_MAP.items():
        entry = providers.get(provider_key, {})
        token = pi_auth._access_token(entry)
        rows.append(
            {
                "provider": provider_key,
                "name": mapping["name"],
                "env": mapping["env"],
                "has_token": bool(token),
                "expired": pi_auth._is_expired(entry) if entry else True,
                "expires": pi_auth._exp_str(entry) if entry else "missing",
            }
        )
    return {"auth_file": str(auth_path), "env_file": str(_env_target_path()), "providers": rows}


def _normalize_terminal_subcommand(raw: str | None) -> str:
    cmd = (raw or "/login").strip().lower()
    if cmd in PI_ALLOWED_COMMANDS:
        return "login"
    raise HTTPException(status_code=400, detail="Only pi /login is allowed")


async def _run_startup_pi_auth_check() -> None:
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(PI_AUTH_SCRIPT),
            "--check",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(Path(__file__).parent.parent),
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=25)
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            log.warning("Startup pi_auth check timed out")
            return
        if proc.returncode != 0:
            log.warning("Startup pi_auth check failed: %s", (err or out).decode(errors="replace").strip())
            return
        text = out.decode(errors="replace").strip()
        if text:
            log.info("pi_auth startup check:\n%s", text)
    except Exception as exc:
        log.warning("Startup pi_auth check skipped: %s", exc)


async def _ws_is_admin(websocket: WebSocket) -> bool:
    session_id = websocket.cookies.get(COOKIE_NAME)
    if not session_id:
        return False
    async with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        row = await db.execute(select(Session).where(Session.id == session_id, Session.expires_at > now))
        sess = row.scalars().first()
        if not sess:
            return False
        user = await db.get(User, sess.user_id)
        return bool(user and user.is_admin)


@app.get("/auth/pi/status")
async def pi_auth_status(_admin=Depends(require_admin)) -> dict:
    auth_path = pi_auth.find_auth_file()
    if not auth_path:
        raise HTTPException(status_code=404, detail="auth.json not found. Run `pi login` first.")
    raw = json.loads(auth_path.read_text(encoding="utf-8"))
    return _token_status_payload(raw, auth_path)


@app.post("/auth/pi/refresh-tokens")
async def pi_auth_refresh(_admin=Depends(require_admin)) -> dict:
    auth_path = pi_auth.find_auth_file()
    if not auth_path:
        raise HTTPException(status_code=404, detail="auth.json not found. Run `pi login` first.")
    providers = pi_auth.load_and_refresh(auth_path, force=False)
    pi_auth.write_to_env(providers, _env_target_path())
    return _token_status_payload(providers, auth_path)


@app.websocket("/terminal")
async def terminal(websocket: WebSocket) -> None:
    if not await _ws_is_admin(websocket):
        await websocket.close(code=1008)
        return

    try:
        subcommand = _normalize_terminal_subcommand(websocket.query_params.get("cmd"))
    except HTTPException:
        await websocket.close(code=1008)
        return

    await websocket.accept()

    try:
        proc = await asyncio.create_subprocess_exec(
            "pi",
            subcommand,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as exc:
        await websocket.send_json({"type": "error", "data": f"Failed to run pi: {exc}"})
        await websocket.close(code=1011)
        return

    async def _pump_output(stream: asyncio.StreamReader | None, stream_name: str) -> None:
        if not stream:
            return
        while True:
            chunk = await stream.read(1024)
            if not chunk:
                break
            await websocket.send_json(
                {"type": "output", "stream": stream_name, "data": chunk.decode(errors="replace")}
            )

    stdout_task = asyncio.create_task(_pump_output(proc.stdout, "stdout"))
    stderr_task = asyncio.create_task(_pump_output(proc.stderr, "stderr"))

    try:
        while True:
            if proc.returncode is not None:
                break
            try:
                message = await asyncio.wait_for(websocket.receive_json(), timeout=0.2)
            except TimeoutError:
                continue
            except WebSocketDisconnect:
                break

            msg_type = str(message.get("type") or "")
            if msg_type == "input" and proc.stdin:
                data = str(message.get("data") or "")
                try:
                    proc.stdin.write(data.encode())
                    await proc.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    break
            elif msg_type == "resize":
                continue
    finally:
        if proc.returncode is None:
            proc.terminate()
        await proc.wait()
        stdout_task.cancel()
        stderr_task.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        try:
            await websocket.send_json({"type": "exit", "code": int(proc.returncode or 0)})
        except Exception:
            pass
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()


# ── Model listing ─────────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models(user=Depends(require_whitelisted)) -> dict:
    return {"object": "list", "data": reg.list_models()}


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
    if entry.provider == "gemini":
        return await gemini_provider.chat(entry.model_id, body, entry.api_key)
    return await openai_compat.chat(
        entry.model_id, body, entry.api_key, entry.base_url, entry.extra_headers
    )


async def _stream(entry: reg.ModelEntry, body: dict) -> AsyncIterator[str]:
    if entry.provider == "gemini":
        async for chunk in gemini_provider.stream(entry.model_id, body, entry.api_key):
            yield chunk
    else:
        async for chunk in openai_compat.stream(
            entry.model_id, body, entry.api_key, entry.base_url, entry.extra_headers
        ):
            yield chunk
