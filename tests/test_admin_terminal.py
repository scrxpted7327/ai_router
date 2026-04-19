"""Tests for admin-gated terminal and auth helpers."""

from __future__ import annotations

import asyncio
import importlib
import uuid

import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException
from starlette.websockets import WebSocketDisconnect
from sqlalchemy import select

from middleware.app import _normalize_terminal_subcommand, app
from middleware.auth import _bearer_digest
from middleware.db import GatewayApiToken, SessionLocal, User

app_module = importlib.import_module("middleware.app")


async def _noop_startup_pi_auth_check() -> None:
    return


def _create_api_token_for_email(email: str) -> str:
    raw_token = f"air_test_{uuid.uuid4().hex}"
    digest = _bearer_digest(raw_token)

    async def _insert() -> None:
        async with SessionLocal() as db:
            user = (await db.execute(select(User).where(User.email == email))).scalars().first()
            if not user:
                raise AssertionError(f"User not found: {email}")
            db.add(GatewayApiToken(user_id=user.id, label="test", token_digest=digest))
            await db.commit()

    asyncio.run(_insert())
    return raw_token


def _set_admin_for_email(email: str, is_admin: bool) -> None:
    async def _update() -> None:
        async with SessionLocal() as db:
            user = (await db.execute(select(User).where(User.email == email))).scalars().first()
            if not user:
                raise AssertionError(f"User not found: {email}")
            user.is_admin = is_admin
            await db.commit()

    asyncio.run(_update())


def test_normalize_terminal_subcommand_accepts_login_variants() -> None:
    assert _normalize_terminal_subcommand("/login") == "login"
    assert _normalize_terminal_subcommand("login") == "login"
    assert _normalize_terminal_subcommand(" LOGIN ") == "login"
    assert _normalize_terminal_subcommand(None) == "login"


def test_normalize_terminal_subcommand_rejects_non_login() -> None:
    try:
        _normalize_terminal_subcommand("/whoami")
        raise AssertionError("Expected HTTPException")
    except HTTPException as exc:
        assert exc.status_code == 400


def test_non_admin_cannot_access_pi_auth_status() -> None:
    email = f"user-{uuid.uuid4().hex[:8]}@example.com"
    password = "test-password-123"
    with TestClient(app) as client:
        register = client.post("/auth/register", json={"email": email, "password": password})
        assert register.status_code == 200
        assert register.json()["is_admin"] is False

        token = _create_api_token_for_email(email)
        status = client.get("/auth/pi/status", headers={"Authorization": f"Bearer {token}"})
        assert status.status_code == 403
        assert status.json()["detail"] == "Admin access required"


def test_auth_me_includes_is_admin_flag() -> None:
    email = f"me-{uuid.uuid4().hex[:8]}@example.com"
    password = "test-password-123"
    with TestClient(app) as client:
        register = client.post("/auth/register", json={"email": email, "password": password})
        assert register.status_code == 200

        token = _create_api_token_for_email(email)
        me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200
        payload = me.json()
        assert payload["email"] == email
        assert payload["is_admin"] is False
        assert "is_whitelisted" in payload


def test_terminal_websocket_rejects_non_admin() -> None:
    email = f"ws-{uuid.uuid4().hex[:8]}@example.com"
    password = "test-password-123"

    with TestClient(app) as client:
        register = client.post("/auth/register", json={"email": email, "password": password})
        assert register.status_code == 200

        login = client.post("/auth/login", json={"email": email, "password": password})
        assert login.status_code == 200

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/terminal?cmd=/login"):
                pass


def test_terminal_websocket_admin_allows_login_and_input(monkeypatch) -> None:
    email = f"admin-{uuid.uuid4().hex[:8]}@example.com"
    password = "test-password-123"
    captured_stdin: list[str] = []

    class _FakeStdin:
        def write(self, data: bytes) -> None:
            captured_stdin.append(data.decode(errors="replace"))

        async def drain(self) -> None:
            return

    class _FakeProcess:
        def __init__(self) -> None:
            self.stdin = _FakeStdin()
            self.stdout = asyncio.StreamReader()
            self.stderr = asyncio.StreamReader()
            self.returncode = None

        def terminate(self) -> None:
            if self.returncode is None:
                self.returncode = 0
            self.stdout.feed_eof()
            self.stderr.feed_eof()

        async def wait(self) -> int:
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

    async def _fake_create_subprocess_exec(*args, **kwargs):
        assert args[0] == "pi"
        assert args[1] == "login"
        proc = _FakeProcess()
        proc.stdout.feed_data(b"Open this localhost URL\n")
        proc.stdout.feed_eof()
        return proc

    async def _always_admin(_websocket) -> bool:
        return True

    monkeypatch.setattr(app_module, "_run_startup_pi_auth_check", _noop_startup_pi_auth_check)
    monkeypatch.setattr(app_module, "_ws_is_admin", _always_admin)
    monkeypatch.setattr(app_module.asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    with TestClient(app) as client:
        with client.websocket_connect("/terminal?cmd=/login") as ws:
            first = ws.receive_json()
            assert first["type"] == "output"
            assert first["stream"] == "stdout"
            assert "localhost" in first["data"]

            ws.send_json({"type": "resize", "cols": 100, "rows": 30})
            ws.send_json({"type": "input", "data": "\x1b[A\x1b[B\rlocalhost:7777\r"})

    sent = "".join(captured_stdin)
    assert "\x1b[A" in sent
    assert "\x1b[B" in sent
    assert "localhost:7777" in sent


def test_terminal_websocket_rejects_non_login_subcommand(monkeypatch) -> None:
    email = f"admin2-{uuid.uuid4().hex[:8]}@example.com"
    password = "test-password-123"

    async def _always_admin(_websocket) -> bool:
        return True

    async def _unexpected(*args, **kwargs):
        raise AssertionError("subprocess should not be started for invalid cmd")

    monkeypatch.setattr(app_module, "_run_startup_pi_auth_check", _noop_startup_pi_auth_check)
    monkeypatch.setattr(app_module, "_ws_is_admin", _always_admin)
    monkeypatch.setattr(app_module.asyncio, "create_subprocess_exec", _unexpected)

    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/terminal?cmd=/whoami"):
                pass
