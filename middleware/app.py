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
from contextlib import asynccontextmanager
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
from .db import ModelControl, Session, SessionLocal, User, init_db
from .format_adapter import normalise_request, stream_as_responses_api
from .router import route
from .providers import anthropic as anthropic_provider
from .providers import bedrock as bedrock_provider
from .providers import gemini as gemini_provider
from .providers import openai_compat
from .tokens import delete_user_token, get_user_token, list_user_tokens, set_user_token

ENV_PATH = Path(__file__).parent.parent / ".env"
STATIC_DASHBOARD = Path(__file__).parent / "static" / "dashboard"
STATIC_TERMINAL = Path(__file__).parent / "static" / "terminal"
PI_AUTH_SCRIPT = Path(__file__).parent.parent / "pi_auth.py"
ALLOWED_CLASSIFICATIONS = {
    "",
    "heavy_reasoning",
    "code_generation",
    "nuanced_coding",
    "multimodal",
    "fast_simple",
}
ALLOWED_EFFORTS = {"default", "low", "medium", "high", "xhigh"}

log = logging.getLogger("ai_router")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "*").strip()
    if raw == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()] or ["*"]


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    await init_db()
    reg.init(str(ENV_PATH))
    log.info("Registry loaded — %d models available", len(reg.list_models()))
    await _run_startup_pi_auth_check()
    yield


app = FastAPI(title="Unified AI Gateway", version="2.0.0", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth_router)
app.include_router(anthropic_router)


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


def _model_classification_guess(model_id: str) -> str:
    text = model_id.lower()
    if any(k in text for k in ("vision", "image", "gemini")):
        return "multimodal"
    if any(k in text for k in ("codex", "code", "dev")):
        return "code_generation"
    if any(k in text for k in ("reason", "r1", "o1", "think", "opus")):
        return "heavy_reasoning"
    if any(k in text for k in ("flash", "instant", "fast", "haiku", "mini")):
        return "fast_simple"
    return "nuanced_coding"


def _model_effort_guess(model_id: str) -> str:
    text = model_id.lower()
    if any(k in text for k in ("flash", "instant", "fast", "mini", "haiku", "8b")):
        return "low"
    if any(k in text for k in ("o1", "o3", "r1", "reason", "70b", "heavy", "xhigh")):
        return "xhigh"
    if any(k in text for k in ("opus", "gpt-5", "high")):
        return "high"
    return "default"


async def _model_control_index() -> tuple[set[str], dict[str, dict[str, str]]]:
    models = reg.list_models()
    model_ids = [m["id"] for m in models]
    async with SessionLocal() as db:
        rows = (
            await db.execute(select(ModelControl).where(ModelControl.model_id.in_(model_ids)))
        ).scalars().all()
    by_id = {row.model_id: row for row in rows}
    enabled: set[str] = set()
    meta: dict[str, dict[str, str]] = {}
    for model_id in model_ids:
        row = by_id.get(model_id)
        row_enabled = bool(row.enabled) if row else True
        if row_enabled:
            enabled.add(model_id)
        meta[model_id] = {
            "classification": (row.classification if row else _model_classification_guess(model_id)) or "",
            "effort": (row.effort if row else _model_effort_guess(model_id)) or "default",
        }
    return enabled, meta


def _target_effort_for_task(task_type: str) -> str:
    if task_type == "heavy_reasoning":
        return "high"
    if task_type == "fast_simple":
        return "low"
    return "default"


def _effort_distance(candidate: str, target: str) -> int:
    order = {"low": 0, "medium": 1, "default": 1, "high": 2, "xhigh": 3}
    return abs(order.get(candidate, 1) - order.get(target, 1))


def _policy_pick_for_task(
    *,
    task_type: str,
    enabled: set[str],
    meta: dict[str, dict[str, str]],
    preferred: list[str],
) -> reg.ModelEntry | None:
    target_effort = _target_effort_for_task(task_type)

    def _is_match(model_id: str) -> bool:
        model_meta = meta.get(model_id, {})
        return model_meta.get("classification", "") == task_type

    candidates = [model_id for model_id in enabled if _is_match(model_id)]
    if not candidates:
        return None

    preferred_pos = {model_id: idx for idx, model_id in enumerate(preferred)}

    candidates.sort(
        key=lambda model_id: (
            _effort_distance(meta.get(model_id, {}).get("effort", "default"), target_effort),
            preferred_pos.get(model_id, 10_000),
            model_id,
        )
    )
    return reg.get(candidates[0])


def _get_terminal_args(raw: str | None) -> list[str]:
    """Get pi arguments. Empty raw means interactive shell."""
    if not raw or not raw.strip():
        return []
    cmd = raw.strip()
    if cmd.startswith("/"):
        cmd = cmd[1:]
    return [cmd] if cmd else []


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


@app.get("/dashboard/model-controls")
async def get_model_controls(_admin=Depends(require_admin)) -> dict:
    models = reg.list_models()
    model_ids = [m["id"] for m in models]
    async with SessionLocal() as db:
        rows = (
            await db.execute(select(ModelControl).where(ModelControl.model_id.in_(model_ids)))
        ).scalars().all()
        existing = {row.model_id: row for row in rows}

        changed = False
        for model_id in model_ids:
            if model_id in existing:
                continue
            row = ModelControl(
                model_id=model_id,
                enabled=True,
                classification=_model_classification_guess(model_id),
                effort=_model_effort_guess(model_id),
            )
            db.add(row)
            existing[model_id] = row
            changed = True
        if changed:
            await db.commit()

    controls = []
    for model in models:
        row = existing[model["id"]]
        controls.append(
            {
                "id": model["id"],
                "name": model.get("name") or model["id"],
                "provider": model.get("owned_by") or "",
                "enabled": bool(row.enabled),
                "classification": row.classification or "",
                "effort": row.effort or "default",
            }
        )
    return {"models": controls}


@app.post("/dashboard/model-controls")
async def set_model_controls(payload: dict, _admin=Depends(require_admin)) -> dict:
    models = payload.get("models")
    if not isinstance(models, list):
        raise HTTPException(status_code=400, detail="models must be a list")

    async with SessionLocal() as db:
        for item in models:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id") or "").strip()
            if not model_id:
                continue

            enabled = bool(item.get("enabled", True))
            classification = str(item.get("classification") or "").strip().lower()
            effort = str(item.get("effort") or "default").strip().lower()

            if classification not in ALLOWED_CLASSIFICATIONS:
                raise HTTPException(status_code=400, detail=f"Invalid classification for {model_id}")
            if effort not in ALLOWED_EFFORTS:
                raise HTTPException(status_code=400, detail=f"Invalid effort for {model_id}")

            row = await db.get(ModelControl, model_id)
            if not row:
                row = ModelControl(model_id=model_id)
                db.add(row)
            row.enabled = enabled
            row.classification = classification
            row.effort = effort

        await db.commit()
    return {"ok": True}


# ── Admin: user + provider-token management ───────────────────────────────────

@app.get("/dashboard/users")
async def admin_list_users(_admin=Depends(require_admin)) -> dict:
    async with SessionLocal() as db:
        from sqlalchemy import func as sqfunc
        from .db import UserProviderToken as UPT
        users = (await db.execute(select(User).order_by(User.created_at))).scalars().all()
        token_counts: dict[str, int] = {}
        for u in users:
            cnt = (await db.execute(
                select(sqfunc.count()).select_from(UPT).where(UPT.user_id == u.id)
            )).scalar() or 0
            token_counts[u.id] = cnt
    return {
        "users": [
            {
                "id": u.id,
                "email": u.email,
                "is_whitelisted": u.is_whitelisted,
                "is_admin": u.is_admin,
                "created_at": u.created_at.isoformat() if u.created_at else None,
                "provider_token_count": token_counts.get(u.id, 0),
            }
            for u in users
        ]
    }


@app.get("/dashboard/users/{user_id}/provider-tokens")
async def admin_get_user_tokens(user_id: str, _admin=Depends(require_admin)) -> dict:
    return {"tokens": await list_user_tokens(user_id)}


@app.put("/dashboard/users/{user_id}/provider-tokens/{provider_id}")
async def admin_set_user_token(
    user_id: str,
    provider_id: str,
    payload: dict,
    _admin=Depends(require_admin),
) -> dict:
    token = str(payload.get("token") or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="token must not be empty")
    await set_user_token(user_id, provider_id, token)
    return {"ok": True}


@app.delete("/dashboard/users/{user_id}/provider-tokens/{provider_id}")
async def admin_delete_user_token(
    user_id: str,
    provider_id: str,
    _admin=Depends(require_admin),
) -> dict:
    deleted = await delete_user_token(user_id, provider_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Token not found")
    return {"ok": True}


@app.websocket("/terminal")
async def terminal(websocket: WebSocket) -> None:
    if not await _ws_is_admin(websocket):
        await websocket.close(code=1008)
        return

    args = _get_terminal_args(websocket.query_params.get("cmd"))
    await websocket.accept()

    try:
        proc = await asyncio.create_subprocess_exec(
            "pi",
            *args,
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
    enabled, meta = await _model_control_index()
    items = []
    for model in reg.list_models():
        if model["id"] not in enabled:
            continue
        entry = dict(model)
        entry.update(meta.get(model["id"], {}))
        items.append(entry)
    return {"object": "list", "data": items}


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

def _auto_route_model(requested: str, task_type: str, enabled: set[str]) -> str | None:
    """Route auto-free/premium/max to actual models based on task classification."""
    AUTO_FREE_ROUTES = {
        "heavy_reasoning": ["deepseek-r1", "qwq-groq", "free"],
        "code_generation": ["deepseek-chat", "qwen", "free"],
        "nuanced_coding": ["deepseek-chat", "qwen", "free"],
        "multimodal": ["llama-3.2-90b-groq", "free-gemma", "free"],
        "fast_simple": ["glm-flash", "free-llama", "free"],
    }
    AUTO_PREMIUM_ROUTES = {
        "heavy_reasoning": ["claude-opus", "o3", "deepseek-r1"],
        "code_generation": ["claude-sonnet", "gpt-4.1", "deepseek-chat"],
        "nuanced_coding": ["claude-sonnet", "gpt-4.1", "gemini-pro"],
        "multimodal": ["gemini-pro", "claude-opus", "gpt-4.1"],
        "fast_simple": ["claude-haiku", "gemini-flash", "cerebras"],
    }
    AUTO_MAX_ROUTES = {
        "heavy_reasoning": ["claude-opus-4-7", "o3", "gpt-5"],
        "code_generation": ["claude-opus-4-7", "gpt-4.1", "o3"],
        "nuanced_coding": ["claude-opus-4-7", "claude-sonnet-4-6", "gpt-4.1"],
        "multimodal": ["gemini-2.5-pro", "claude-opus-4-7", "gpt-4.1"],
        "fast_simple": ["claude-sonnet-4-6", "gpt-4.1", "gemini-2.5-flash"],
    }

    routes_map = {
        "scrxpted/auto-free": AUTO_FREE_ROUTES,
        "auto-free": AUTO_FREE_ROUTES,
        "scrxpted/auto-premium": AUTO_PREMIUM_ROUTES,
        "auto-premium": AUTO_PREMIUM_ROUTES,
        "scrxpted/auto-max": AUTO_MAX_ROUTES,
        "auto-max": AUTO_MAX_ROUTES,
    }

    route_set = routes_map.get(requested)
    if not route_set:
        return None

    candidates = route_set.get(task_type, route_set.get("nuanced_coding", []))
    for candidate in candidates:
        entry = reg.get(candidate)
        if entry and entry.model_id in enabled:
            return entry.model_id
    return None


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
    enabled, meta = await _model_control_index()
    entry = reg.get(requested)

    # Handle auto-routing pseudo-models
    if requested and entry and entry.base_url == "INTERNAL":
        decision = route(messages)
        task_type = getattr(decision.task_type, "value", str(decision.task_type))
        auto_model_id = _auto_route_model(requested, task_type, enabled)
        if auto_model_id:
            entry = reg.get(auto_model_id)
            log.info("Auto-route '%s' [%s] → %s", requested, task_type, auto_model_id)
        else:
            raise HTTPException(status_code=503, detail=f"No models available for auto-routing tier '{requested}'")
    elif requested and requested not in enabled:
        raise HTTPException(status_code=403, detail=f"Model '{requested}' is disabled by admin policy")
    if not entry:
        decision = route(messages)
        task_type = getattr(decision.task_type, "value", str(decision.task_type))
        primary = decision.primary if decision.primary in enabled else ""
        entry = reg.get(primary) if primary else None
        if not entry:
            # walk fallbacks
            for fb in decision.fallbacks:
                if fb not in enabled:
                    continue
                entry = reg.get(fb)
                if entry:
                    break
        if not entry:
            entry = _policy_pick_for_task(
                task_type=task_type,
                enabled=enabled,
                meta=meta,
                preferred=[decision.primary, *decision.fallbacks],
            )
        if not entry:
            raise HTTPException(status_code=503, detail=f"No provider available for '{requested}'")
        log.info("Auto-routed '%s' → %s (%s)", requested, entry.model_id, entry.provider)
    else:
        log.info("Model '%s' → %s (%s)", requested, entry.model_id, entry.provider)

    # 3. Apply user's personal API key if one is stored; non-admins must have one
    if user and entry.provider_id:
        user_key = await get_user_token(user.id, entry.provider_id)
        if user_key:
            from dataclasses import replace as dc_replace
            entry = dc_replace(entry, api_key=user_key)
            log.info("Using personal token for provider '%s'", entry.provider_id)
        elif not user.is_admin:
            raise HTTPException(
                status_code=403,
                detail=f"No personal API key for provider '{entry.provider_id}'. Add one in Dashboard → API Keys.",
            )

    if entry.provider != "bedrock" and (not entry.api_key or "REPLACE_ME" in entry.api_key):
        raise HTTPException(status_code=401,
                            detail=f"No API key for '{entry.provider}'. Run `python pi_auth.py`.")

    # 4. Dispatch to provider
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
    if entry.provider == "bedrock":
        return await bedrock_provider.chat(entry.model_id, body, entry.options)
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
    elif entry.provider == "bedrock":
        async for chunk in bedrock_provider.stream(entry.model_id, body, entry.options):
            yield chunk
    else:
        async for chunk in openai_compat.stream(
            entry.model_id, body, entry.api_key, entry.base_url, entry.extra_headers
        ):
            yield chunk
