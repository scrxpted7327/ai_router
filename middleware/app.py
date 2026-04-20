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
from datetime import datetime, timedelta, timezone
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
from .db import (
    AutoRouterConfig, AutoRouterConfigHistory, ModelControl, ProviderSetting,
    RouteAnalytics, Session, SessionLocal, User, UserRoutingPreference, init_db,
)
from .format_adapter import normalise_request, stream_as_responses_api
from .router import route, TaskType, _ROUTES
from .providers import ProviderRateLimitError
from .providers import anthropic as anthropic_provider
from .providers import bedrock as bedrock_provider
from .providers import gemini as gemini_provider
from .providers import openai_compat
from .providers import pi_cli as pi_cli_provider
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
    model_provider = {m["id"]: m.get("provider_id", "") for m in models}
    async with SessionLocal() as db:
        mc_rows = (
            await db.execute(select(ModelControl).where(ModelControl.model_id.in_(model_ids)))
        ).scalars().all()
        ps_rows = (await db.execute(select(ProviderSetting))).scalars().all()
    by_id = {row.model_id: row for row in mc_rows}
    disabled_providers: set[str] = set()
    for ps in ps_rows:
        s = json.loads(ps.settings_json or "{}")
        if not s.get("enabled", True):
            disabled_providers.add(ps.provider_id)
    enabled: set[str] = set()
    meta: dict[str, dict[str, str]] = {}
    for model_id in model_ids:
        row = by_id.get(model_id)
        provider_id = model_provider.get(model_id, "")
        row_enabled = bool(row.enabled) if row else True
        provider_enabled = provider_id not in disabled_providers
        if row_enabled and provider_enabled:
            enabled.add(model_id)
        meta[model_id] = {
            "classification": (row.classification if row else _model_classification_guess(model_id)) or "",
            "effort": (row.effort if row else _model_effort_guess(model_id)) or "default",
            "pinned": bool(row.pinned) if row else False,
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
            0 if meta.get(model_id, {}).get("pinned") else 1,
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
                pinned=False,
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
                "pinned": bool(row.pinned),
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
            pinned = bool(item.get("pinned", False))
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
            row.pinned = pinned
            row.classification = classification
            row.effort = effort

        await db.commit()
    return {"ok": True}


@app.get("/dashboard/auto-router-config")
async def get_auto_router_config(_admin=Depends(require_admin)) -> dict:
    """Get auto-router configuration (scrxpted/auto-light, auto-free, auto-premium, auto-max)."""
    tiers = ["auto-light", "auto-free", "auto-premium", "auto-max"]
    task_types = ["heavy_reasoning", "code_generation", "nuanced_coding", "multimodal", "fast_simple"]

    async with SessionLocal() as db:
        configs_list = (await db.execute(select(AutoRouterConfig))).scalars().all()
        configs_by_id = {c.id: c for c in configs_list}

    result = {}
    for tier in tiers:
        result[tier] = {}
        for task_type in task_types:
            config_id = f"{tier}:{task_type}"
            config = configs_by_id.get(config_id)
            if config and config.model_ids:
                try:
                    result[tier][task_type] = json.loads(config.model_ids)
                except (json.JSONDecodeError, TypeError):
                    result[tier][task_type] = []
            else:
                result[tier][task_type] = []

    return {"configs": result}


@app.post("/dashboard/auto-router-config")
async def set_auto_router_config(payload: dict, admin=Depends(require_admin)) -> dict:
    """Set auto-router configuration. payload: {configs: {tier: {task_type: [model_ids]}}}"""
    configs = payload.get("configs")
    if not isinstance(configs, dict):
        raise HTTPException(status_code=400, detail="configs must be a dict")

    async with SessionLocal() as db:
        for tier, task_map in configs.items():
            if not isinstance(task_map, dict):
                continue
            for task_type, model_ids in task_map.items():
                if not isinstance(model_ids, list):
                    continue

                config_id = f"{tier}:{task_type}"
                row = (
                    await db.execute(select(AutoRouterConfig).where(AutoRouterConfig.id == config_id))
                ).scalars().first()

                if not row:
                    row = AutoRouterConfig(id=config_id, tier=tier, task_type=task_type)
                    db.add(row)

                row.model_ids = json.dumps(model_ids)

                db.add(AutoRouterConfigHistory(
                    config_id=config_id,
                    tier=tier,
                    task_type=task_type,
                    model_ids=json.dumps(model_ids),
                    changed_by=admin.id if admin else None,
                ))

        await db.commit()

    return {"ok": True}


@app.get("/dashboard/provider-settings")
async def get_provider_settings(_admin=Depends(require_admin)) -> dict:
    """Get provider-specific settings."""
    async with SessionLocal() as db:
        rows = (await db.execute(select(ProviderSetting))).scalars().all()

    settings = {}
    for row in rows:
        try:
            settings[row.provider_id] = json.loads(row.settings_json)
        except (json.JSONDecodeError, TypeError):
            settings[row.provider_id] = {}

    return {"settings": settings}


@app.post("/dashboard/provider-settings")
async def set_provider_settings(payload: dict, _admin=Depends(require_admin)) -> dict:
    """Set provider-specific settings. payload: {settings: {provider_id: {key: value}}}"""
    settings = payload.get("settings")
    if not isinstance(settings, dict):
        raise HTTPException(status_code=400, detail="settings must be a dict")

    async with SessionLocal() as db:
        for provider_id, provider_settings in settings.items():
            if not isinstance(provider_settings, dict):
                continue

            row = await db.get(ProviderSetting, provider_id)
            if not row:
                row = ProviderSetting(provider_id=provider_id)
                db.add(row)

            row.settings_json = json.dumps(provider_settings)

        await db.commit()

    return {"ok": True}


@app.get("/dashboard/providers")
async def get_providers(_admin=Depends(require_admin)) -> dict:
    """List all registry providers with enabled/disabled state."""
    models = reg.list_models()
    # Count canonical models per provider
    provider_counts: dict[str, int] = {}
    for m in models:
        pid = m.get("provider_id", "")
        if pid:
            provider_counts[pid] = provider_counts.get(pid, 0) + 1

    async with SessionLocal() as db:
        ps_rows = (await db.execute(select(ProviderSetting))).scalars().all()
    ps_by_id = {row.provider_id: json.loads(row.settings_json or "{}") for row in ps_rows}

    # Build provider list from registry
    seen: set[str] = set()
    providers = []
    for m in models:
        pid = m.get("provider_id", "")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        s = ps_by_id.get(pid, {})
        providers.append({
            "id": pid,
            "label": m.get("owned_by") or pid,
            "api": m.get("provider_api") or "",
            "enabled": s.get("enabled", True),
            "model_count": provider_counts.get(pid, 0),
        })

    providers.sort(key=lambda p: p["label"].lower())
    return {"providers": providers}


@app.post("/dashboard/providers")
async def set_providers(payload: dict, _admin=Depends(require_admin)) -> dict:
    """Enable/disable providers. payload: {providers: {provider_id: {enabled: bool}}}"""
    updates = payload.get("providers")
    if not isinstance(updates, dict):
        raise HTTPException(status_code=400, detail="providers must be a dict")

    async with SessionLocal() as db:
        for provider_id, settings in updates.items():
            if not isinstance(settings, dict):
                continue
            row = await db.get(ProviderSetting, provider_id)
            if not row:
                row = ProviderSetting(provider_id=provider_id, settings_json="{}")
                db.add(row)
            existing = json.loads(row.settings_json or "{}")
            if "enabled" in settings:
                existing["enabled"] = bool(settings["enabled"])
            row.settings_json = json.dumps(existing)
        await db.commit()

    return {"ok": True}


# ── Routing: preferences, preview, validation, analytics, import/export ──────

@app.get("/api/routing/preferences")
async def get_routing_preferences(user=Depends(require_whitelisted)) -> dict:
    async with SessionLocal() as db:
        pref = (
            await db.execute(
                select(UserRoutingPreference).where(UserRoutingPreference.user_id == user.id)
            )
        ).scalars().first()
    if not pref:
        return {
            "preferred_models": {},
            "avoid_models": [],
            "tier_overrides": {},
            "provider_priority": [],
            "enabled": True,
        }
    def _safe_json(raw: str | None, default):
        try:
            return json.loads(raw or "") if raw else default
        except (json.JSONDecodeError, TypeError):
            return default

    return {
        "preferred_models": _safe_json(pref.preferred_models, {}),
        "avoid_models": _safe_json(pref.avoid_models, []),
        "tier_overrides": _safe_json(pref.tier_overrides, {}),
        "provider_priority": _safe_json(pref.provider_priority, []),
        "enabled": pref.enabled,
    }


@app.put("/api/routing/preferences")
async def update_routing_preferences(payload: dict, user=Depends(require_whitelisted)) -> dict:
    preferred_models = payload.get("preferred_models", {})
    avoid_models = payload.get("avoid_models", [])
    tier_overrides = payload.get("tier_overrides", {})
    provider_priority = payload.get("provider_priority", [])
    enabled = payload.get("enabled", True)

    if not isinstance(preferred_models, dict):
        raise HTTPException(status_code=400, detail="preferred_models must be a dict")
    if not isinstance(avoid_models, list):
        raise HTTPException(status_code=400, detail="avoid_models must be a list")
    if not isinstance(tier_overrides, dict):
        raise HTTPException(status_code=400, detail="tier_overrides must be a dict")
    if not isinstance(provider_priority, list):
        raise HTTPException(status_code=400, detail="provider_priority must be a list")

    async with SessionLocal() as db:
        pref = (
            await db.execute(
                select(UserRoutingPreference).where(UserRoutingPreference.user_id == user.id)
            )
        ).scalars().first()
        if not pref:
            pref = UserRoutingPreference(user_id=user.id)
            db.add(pref)
        pref.preferred_models = json.dumps(preferred_models)
        pref.avoid_models = json.dumps(avoid_models)
        pref.tier_overrides = json.dumps(tier_overrides)
        pref.provider_priority = json.dumps(provider_priority)
        pref.enabled = bool(enabled)
        await db.commit()

    return {"ok": True}


@app.post("/api/routing/preview")
async def preview_routing(payload: dict, user=Depends(require_whitelisted)) -> dict:
    task_type = str(payload.get("task_type", "nuanced_coding")).strip()
    tier = str(payload.get("tier", "scrxpted/auto-premium")).strip()
    if task_type not in ALLOWED_CLASSIFICATIONS or not task_type:
        raise HTTPException(status_code=400, detail=f"Invalid task_type: {task_type}")

    enabled, _meta = await _model_control_index()
    model_id = await _auto_route_model(tier, task_type, enabled, user=user)
    if not model_id:
        return {"selected_model": None, "provider": None, "reason": "no_match"}

    entry = reg.get(model_id)
    return {
        "selected_model": model_id,
        "provider": entry.provider_id if entry else None,
        "reason": "routed",
    }


@app.post("/api/routing/validate")
async def validate_model_ids(payload: dict, _admin=Depends(require_admin)) -> dict:
    model_ids = payload.get("model_ids", [])
    if not isinstance(model_ids, list):
        raise HTTPException(status_code=400, detail="model_ids must be a list")

    valid, invalid, warnings = [], [], []
    for mid in model_ids:
        mid = str(mid).strip()
        if not mid:
            continue
        entry = reg.get(mid)
        if entry:
            valid.append({"id": mid, "resolved": entry.model_id, "provider": entry.provider_id})
        else:
            invalid.append(mid)
            if "/" in mid:
                parts = mid.split("/", 1)
                bare_entry = reg.get(parts[1])
                if bare_entry:
                    warnings.append(
                        f"'{mid}' not found, but '{parts[1]}' exists via {bare_entry.provider_id}"
                    )
    return {"valid": valid, "invalid": invalid, "warnings": warnings}


@app.get("/api/routing/analytics")
async def get_routing_analytics(
    _admin=Depends(require_admin),
    days: int = 7,
    task_type: str | None = None,
) -> dict:
    from sqlalchemy import func as sqfunc

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    async with SessionLocal() as db:
        q = select(RouteAnalytics).where(RouteAnalytics.timestamp >= cutoff)
        if task_type:
            q = q.where(RouteAnalytics.task_type == task_type)
        rows = (await db.execute(q.order_by(RouteAnalytics.timestamp.desc()).limit(5000))).scalars().all()

    tier_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    model_counts: dict[str, int] = {}
    provider_counts: dict[str, int] = {}
    user_pref_count = 0

    for r in rows:
        tier_counts[r.tier or "direct"] = tier_counts.get(r.tier or "direct", 0) + 1
        task_counts[r.task_type or "unknown"] = task_counts.get(r.task_type or "unknown", 0) + 1
        model_counts[r.selected_model] = model_counts.get(r.selected_model, 0) + 1
        provider_counts[r.selected_provider or "unknown"] = provider_counts.get(r.selected_provider or "unknown", 0) + 1
        if r.user_preference_applied:
            user_pref_count += 1

    return {
        "total": len(rows),
        "period_days": days,
        "tier_distribution": tier_counts,
        "task_distribution": task_counts,
        "top_models": dict(sorted(model_counts.items(), key=lambda x: -x[1])[:20]),
        "provider_distribution": provider_counts,
        "user_preference_rate": round(user_pref_count / max(len(rows), 1) * 100, 1),
    }


@app.get("/api/routing/history")
async def get_routing_history(_admin=Depends(require_admin), limit: int = 50) -> dict:
    async with SessionLocal() as db:
        rows = (
            await db.execute(
                select(AutoRouterConfigHistory)
                .order_by(AutoRouterConfigHistory.timestamp.desc())
                .limit(min(limit, 200))
            )
        ).scalars().all()

    return {
        "history": [
            {
                "id": r.id,
                "config_id": r.config_id,
                "tier": r.tier,
                "task_type": r.task_type,
                "model_ids": json.loads(r.model_ids or "[]"),
                "changed_by": r.changed_by,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            }
            for r in rows
        ]
    }


@app.get("/api/routing/export")
async def export_routing_config(_admin=Depends(require_admin)) -> dict:
    async with SessionLocal() as db:
        configs = (await db.execute(select(AutoRouterConfig))).scalars().all()
    result = {}
    for c in configs:
        result[c.id] = {
            "tier": c.tier,
            "task_type": c.task_type,
            "model_ids": json.loads(c.model_ids or "[]"),
        }
    return {"configs": result}


@app.post("/api/routing/import")
async def import_routing_config(payload: dict, admin=Depends(require_admin)) -> dict:
    configs = payload.get("configs")
    if not isinstance(configs, dict):
        raise HTTPException(status_code=400, detail="configs must be a dict")

    imported = 0
    async with SessionLocal() as db:
        for config_id, data in configs.items():
            if not isinstance(data, dict):
                continue
            tier = str(data.get("tier", "")).strip()
            task_type = str(data.get("task_type", "")).strip()
            model_ids = data.get("model_ids", [])
            if not tier or not task_type or not isinstance(model_ids, list):
                continue

            row = (
                await db.execute(select(AutoRouterConfig).where(AutoRouterConfig.id == config_id))
            ).scalars().first()
            if not row:
                row = AutoRouterConfig(id=config_id, tier=tier, task_type=task_type)
                db.add(row)
            row.model_ids = json.dumps(model_ids)

            db.add(AutoRouterConfigHistory(
                config_id=config_id,
                tier=tier,
                task_type=task_type,
                model_ids=json.dumps(model_ids),
                changed_by=admin.id if admin else None,
            ))
            imported += 1

        await db.commit()

    return {"ok": True, "imported": imported}


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

    # Send welcome message
    await websocket.send_json({"type": "output", "stream": "stdout", "data": "Starting pi CLI with PTY...\r\n"})

    try:
        import ptyprocess
        import os as os_module
        import struct
        import fcntl
        import termios

        # Spawn pi with a real PTY
        cmd_args = ["pi"] + args if args else ["pi"]
        pty = ptyprocess.PtyProcess.spawn(cmd_args, dimensions=(24, 80))

        await websocket.send_json({"type": "output", "stream": "stdout", "data": "PTY started. Type commands and press Enter.\r\n\r\n"})
    except Exception as exc:
        await websocket.send_json({"type": "error", "data": f"Failed to start PTY: {exc}"})
        await websocket.close(code=1011)
        return

    async def _read_output():
        """Read from PTY and send to WebSocket"""
        while pty.isalive():
            try:
                # Non-blocking read with timeout
                data = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: pty.read(1024) if pty.isalive() else None
                )
                if data:
                    await websocket.send_json({"type": "output", "stream": "stdout", "data": data.decode(errors="replace")})
                else:
                    await asyncio.sleep(0.05)
            except EOFError:
                break
            except Exception as e:
                log.warning(f"PTY read error: {e}")
                break

    read_task = asyncio.create_task(_read_output())

    try:
        while pty.isalive():
            try:
                message = await asyncio.wait_for(websocket.receive_json(), timeout=0.2)
            except TimeoutError:
                continue
            except WebSocketDisconnect:
                break

            msg_type = str(message.get("type") or "")
            if msg_type == "input":
                data = str(message.get("data") or "")
                try:
                    pty.write(data.encode())
                except (BrokenPipeError, OSError):
                    break
            elif msg_type == "resize":
                cols = message.get("cols", 80)
                rows = message.get("rows", 24)
                try:
                    pty.setwinsize(rows, cols)
                except Exception:
                    pass
    finally:
        read_task.cancel()
        try:
            await asyncio.gather(read_task, return_exceptions=True)
        except Exception:
            pass

        if pty.isalive():
            pty.terminate()
            try:
                pty.wait()
            except Exception:
                pass

        try:
            await websocket.send_json({"type": "exit", "code": pty.exitstatus or 0})
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

def _resolve_with_provider_priority(
    model_name: str,
    provider_priority: list[str],
    enabled: set[str],
) -> str | None:
    if not model_name or not isinstance(model_name, str):
        return None

    if "/" in model_name:
        entry = reg.get(model_name)
        if entry and entry.model_id in enabled:
            return entry.model_id

    for provider_id in provider_priority:
        if not provider_id:
            continue
        entry = reg.get_from_provider(provider_id, model_name)
        if entry and entry.model_id in enabled:
            return entry.model_id

    entry = reg.get(model_name)
    return entry.model_id if entry and entry.model_id in enabled else None


async def _log_route_analytics(
    user_id: str | None,
    requested_model: str,
    task_type: str,
    selected_model: str,
    selected_provider: str,
    tier: str,
    user_preference_applied: bool,
    fallback_count: int = 0,
) -> None:
    try:
        async with SessionLocal() as db:
            db.add(RouteAnalytics(
                user_id=user_id,
                requested_model=requested_model,
                task_type=task_type,
                selected_model=selected_model,
                selected_provider=selected_provider,
                tier=tier,
                user_preference_applied=user_preference_applied,
                fallback_count=fallback_count,
            ))
            await db.commit()
    except Exception:
        log.debug("Failed to log route analytics", exc_info=True)


async def _auto_route_model(
    requested: str,
    task_type: str,
    enabled: set[str],
    user: User | None = None,
) -> str | None:
    """Route auto-light/free/premium/max to actual models based on task classification."""
    AUTO_LIGHT_ROUTES = {
        "heavy_reasoning": ["qwq-groq", "deepseek-r1-groq", "llama-3.3-70b"],
        "code_generation": ["kimi-groq", "llama-3.3-70b", "qwen3-groq"],
        "nuanced_coding":  ["llama-3.3-70b", "kimi-groq", "qwen3-groq"],
        "multimodal":      ["llama-4-maverick-groq", "llama-3.2-90b-groq", "llama-3.2-11b-groq"],
        "fast_simple":     ["llama-fast", "gemma2-groq", "llama3.1-8b-cerebras"],
    }
    AUTO_FREE_ROUTES = {
        "heavy_reasoning": ["big-pickle", "nemotron-free", "deepseek-r1-groq"],
        "code_generation": ["big-pickle", "nemotron-free", "minimax-free"],
        "nuanced_coding":  ["big-pickle", "minimax-free", "nemotron-free"],
        "multimodal":      ["minimax-free", "free-gemma", "big-pickle"],
        "fast_simple":     ["minimax-free", "trinity-free", "nemotron-free"],
    }
    AUTO_PREMIUM_ROUTES = {
        "heavy_reasoning": ["github-copilot-pi/claude-sonnet-4-6", "openai-codex/gpt-5.3-codex", "deepseek-r1-groq"],
        "code_generation": ["openai-codex/gpt-5.3-codex", "github-copilot-pi/gpt-5.3-codex", "github-copilot-pi/claude-sonnet-4-6"],
        "nuanced_coding":  ["github-copilot-pi/claude-sonnet-4-6", "openai-codex/gpt-5.3-codex", "deepseek-r1-groq"],
        "multimodal":      ["google-antigravity/gemini-3-pro-preview", "google-gemini-cli/gemini-3-pro-preview", "google-gemini-cli/gemini-3-flash-preview"],
        "fast_simple":     ["github-copilot-pi/claude-haiku-4-5-20251001", "zen-haiku", "cerebras-fast"],
    }
    AUTO_MAX_ROUTES = {
        "heavy_reasoning": ["github-copilot-pi/claude-opus-4-7", "google-antigravity/claude-opus-4-7", "openai-codex/gpt-5.4"],
        "code_generation": ["github-copilot-pi/claude-opus-4-7", "openai-codex/gpt-5.4", "github-copilot-pi/claude-sonnet-4-6"],
        "nuanced_coding":  ["github-copilot-pi/claude-opus-4-7", "github-copilot-pi/claude-sonnet-4-6", "openai-codex/gpt-5.4"],
        "multimodal":      ["google-antigravity/gemini-3.1-pro-preview", "google-gemini-cli/gemini-3.1-pro-preview", "github-copilot-pi/claude-opus-4-7"],
        "fast_simple":     ["github-copilot-pi/claude-sonnet-4-6", "github-copilot-pi/claude-opus-4-7", "openai-codex/gpt-5.4"],
    }

    routes_map = {
        "scrxpted/auto-light": AUTO_LIGHT_ROUTES,
        "auto-light": AUTO_LIGHT_ROUTES,
        "scrxpted/auto-free": AUTO_FREE_ROUTES,
        "auto-free": AUTO_FREE_ROUTES,
        "scrxpted/auto-premium": AUTO_PREMIUM_ROUTES,
        "auto-premium": AUTO_PREMIUM_ROUTES,
        "scrxpted/auto-max": AUTO_MAX_ROUTES,
        "auto-max": AUTO_MAX_ROUTES,
    }

    tier = requested.replace("scrxpted/", "")

    # Priority 1: User routing preferences
    if user:
        async with SessionLocal() as db:
            pref = (
                await db.execute(
                    select(UserRoutingPreference).where(UserRoutingPreference.user_id == user.id)
                )
            ).scalars().first()

        if pref and pref.enabled:
            try:
                avoid = set(json.loads(pref.avoid_models or "[]"))
                provider_priority = json.loads(pref.provider_priority or "[]")
                tier_overrides = json.loads(pref.tier_overrides or "{}")
                preferred_models = json.loads(pref.preferred_models or "{}")
            except (json.JSONDecodeError, TypeError):
                avoid, provider_priority, tier_overrides, preferred_models = set(), [], {}, {}

            if tier in tier_overrides:
                override = tier_overrides[tier]
                if override and isinstance(override, str):
                    resolved = _resolve_with_provider_priority(override, provider_priority, enabled)
                    if resolved and resolved not in avoid:
                        entry = reg.get(resolved)
                        if entry:
                            asyncio.create_task(_log_route_analytics(
                                user.id, requested, task_type, resolved, entry.provider_id, tier, True,
                            ))
                        return resolved

            if task_type in preferred_models:
                task_models = preferred_models[task_type]
                if isinstance(task_models, list):
                    for idx, model in enumerate(task_models):
                        resolved = _resolve_with_provider_priority(model, provider_priority, enabled)
                        if resolved and resolved not in avoid:
                            entry = reg.get(resolved)
                            if entry:
                                asyncio.create_task(_log_route_analytics(
                                    user.id, requested, task_type, resolved, entry.provider_id, tier, True, idx,
                                ))
                            return resolved

    # Priority 2: Admin AutoRouterConfig from database
    async with SessionLocal() as db:
        config_id = f"{tier}:{task_type}"
        config = (
            await db.execute(select(AutoRouterConfig).where(AutoRouterConfig.id == config_id))
        ).scalars().first()

        if config and config.model_ids:
            try:
                candidates = json.loads(config.model_ids)
                if isinstance(candidates, list) and candidates:
                    for idx, candidate in enumerate(candidates):
                        entry = reg.get(candidate)
                        if entry and entry.model_id in enabled:
                            if user:
                                asyncio.create_task(_log_route_analytics(
                                    user.id, requested, task_type, entry.model_id, entry.provider_id, tier, False, idx,
                                ))
                            return f"{entry.provider_id}/{entry.model_id}"
            except (json.JSONDecodeError, TypeError):
                pass

    # Priority 3: Hardcoded fallback routes
    route_set = routes_map.get(tier) or routes_map.get(requested)
    if not route_set:
        return None

    candidates = route_set.get(task_type, route_set.get("nuanced_coding", []))
    for idx, candidate in enumerate(candidates):
        entry = reg.get(candidate)
        if entry and entry.model_id in enabled:
            if user:
                asyncio.create_task(_log_route_analytics(
                    user.id, requested, task_type, entry.model_id, entry.provider_id, tier, False, idx,
                ))
            return f"{entry.provider_id}/{entry.model_id}"
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
        auto_model_id = await _auto_route_model(requested, task_type, enabled, user=user)
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
    # pi_cli providers use server-side auth — no per-user key needed
    if user and entry.provider_id and entry.provider != "pi_cli":
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

    if entry.provider not in ("bedrock", "pi_cli") and (not entry.api_key or "REPLACE_ME" in entry.api_key):
        raise HTTPException(status_code=401,
                            detail=f"No API key for '{entry.provider}'. Run `python pi_auth.py`.")

    # 4. Dispatch with automatic rate-limit fallback
    task_type_str = locals().get("task_type") or getattr(route(messages).task_type, "value", "nuanced_coding")
    tried_models: set[str] = {entry.model_id}

    for _attempt in range(4):  # up to 3 fallbacks
        try:
            if do_stream:
                return await _try_stream(entry, body, is_responses_api)
            else:
                result = await _complete(entry, body)
                return Response(content=json.dumps(result), media_type="application/json")
        except ProviderRateLimitError:
            tried_models.add(entry.model_id)
            log.warning("Rate limit on %s (attempt %d), switching model", entry.model_id, _attempt + 1)
            next_entry = _rl_fallback(task_type_str, enabled, tried_models)
            if not next_entry:
                raise HTTPException(status_code=429,
                                    detail="All candidate models are rate-limited. Try again later.")
            log.info("Rate-limit fallback: %s → %s", entry.model_id, next_entry.model_id)
            entry = next_entry

    raise HTTPException(status_code=429, detail="All candidate models are rate-limited.")


def _rl_fallback(task_type: str, enabled: set[str], tried: set[str]) -> reg.ModelEntry | None:
    """Return the next untried model for the given task type using the router's fallback chain."""
    try:
        decision = _ROUTES[TaskType(task_type)]
    except (ValueError, KeyError):
        return None
    for model_id in [decision.primary, *decision.fallbacks]:
        if model_id in tried:
            continue
        entry = reg.get(model_id)
        if entry and entry.model_id in enabled and entry.model_id not in tried:
            return entry
    return None


async def _try_stream(entry: reg.ModelEntry, body: dict, is_responses_api: bool) -> Response:
    """Begin streaming, eagerly fetching the first chunk to surface rate-limit errors early."""
    gen = _stream(entry, body)
    try:
        first_chunk = await gen.__anext__()
    except StopAsyncIteration:
        first_chunk = None

    async def _combined() -> AsyncIterator[str]:
        if first_chunk is not None:
            yield first_chunk
        async for chunk in gen:
            yield chunk

    raw_stream: AsyncIterator[str] = _combined()
    if is_responses_api:
        raw_stream = stream_as_responses_api(raw_stream, entry.model_id)
    return StreamingResponse(raw_stream, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def _complete(entry: reg.ModelEntry, body: dict) -> dict:
    try:
        if entry.provider == "anthropic":
            return await anthropic_provider.chat(entry.model_id, body, entry.api_key)
        if entry.provider == "gemini":
            return await gemini_provider.chat(entry.model_id, body, entry.api_key)
        if entry.provider == "bedrock":
            return await bedrock_provider.chat(entry.model_id, body, entry.options)
        if entry.provider == "pi_cli":
            return await pi_cli_provider.chat(entry.model_id, body)
        return await openai_compat.chat(
            entry.model_id, body, entry.api_key, entry.base_url, entry.extra_headers
        )
    except Exception as exc:
        _maybe_raise_rl(exc)
        raise


async def _stream(entry: reg.ModelEntry, body: dict) -> AsyncIterator[str]:
    try:
        if entry.provider == "anthropic":
            async for chunk in anthropic_provider.stream(entry.model_id, body, entry.api_key):
                yield chunk
        elif entry.provider == "gemini":
            async for chunk in gemini_provider.stream(entry.model_id, body, entry.api_key):
                yield chunk
        elif entry.provider == "bedrock":
            async for chunk in bedrock_provider.stream(entry.model_id, body, entry.options):
                yield chunk
        elif entry.provider == "pi_cli":
            async for chunk in pi_cli_provider.stream(entry.model_id, body):
                yield chunk
        else:
            async for chunk in openai_compat.stream(
                entry.model_id, body, entry.api_key, entry.base_url, entry.extra_headers
            ):
                yield chunk
    except Exception as exc:
        _maybe_raise_rl(exc)
        raise


def _maybe_raise_rl(exc: Exception) -> None:
    """Convert provider-specific rate-limit exceptions to ProviderRateLimitError."""
    name = type(exc).__name__
    msg = str(exc)
    if (
        "RateLimitError" in name
        or "ResourceExhausted" in name
        or "429" in msg
        or "rate limit" in msg.lower()
        or "too many requests" in msg.lower()
    ):
        raise ProviderRateLimitError(msg) from exc
