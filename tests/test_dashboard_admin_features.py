"""Tests for dashboard admin controls and token endpoints."""

from __future__ import annotations

import asyncio
import uuid

from fastapi.testclient import TestClient
from sqlalchemy import select

from middleware.app import app
from middleware.auth import _bearer_digest
from middleware.db import GatewayApiToken, SessionLocal, User


def _make_user(email: str, password: str) -> None:
    with TestClient(app) as client:
        r = client.post("/auth/register", json={"email": email, "password": password})
        assert r.status_code == 200


def _set_admin(email: str, is_admin: bool = True, is_whitelisted: bool | None = None) -> None:
    async def _run() -> None:
        async with SessionLocal() as db:
            user = (await db.execute(select(User).where(User.email == email))).scalars().first()
            assert user is not None
            user.is_admin = is_admin
            if is_whitelisted is not None:
                user.is_whitelisted = is_whitelisted
            await db.commit()

    asyncio.run(_run())


def _create_token(email: str) -> str:
    raw_token = f"air_test_{uuid.uuid4().hex}"
    digest = _bearer_digest(raw_token)

    async def _run() -> None:
        async with SessionLocal() as db:
            user = (await db.execute(select(User).where(User.email == email))).scalars().first()
            assert user is not None
            db.add(GatewayApiToken(user_id=user.id, label="test", token_digest=digest))
            await db.commit()

    asyncio.run(_run())
    return raw_token


def test_token_regenerate_returns_new_bearer() -> None:
    email = f"tok-{uuid.uuid4().hex[:8]}@example.com"
    password = "test-password-123"
    _make_user(email, password)

    with TestClient(app) as client:
        old = _create_token(email)
        meta = client.get("/auth/tokens", headers={"Authorization": f"Bearer {old}"})
        assert meta.status_code == 200
        assert meta.json()["token_count"] >= 1

        regen = client.post("/auth/tokens/regenerate", headers={"Authorization": f"Bearer {old}"})
        assert regen.status_code == 200
        token = regen.json().get("token")
        assert isinstance(token, str)
        assert token.startswith("air_")

        old_invalid = client.get("/auth/me", headers={"Authorization": f"Bearer {old}"})
        assert old_invalid.status_code == 401

        new_ok = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert new_ok.status_code == 200


def test_model_controls_admin_only() -> None:
    email = f"controls-{uuid.uuid4().hex[:8]}@example.com"
    password = "test-password-123"
    _make_user(email, password)
    token = _create_token(email)

    with TestClient(app) as client:
        denied = client.get("/dashboard/model-controls", headers={"Authorization": f"Bearer {token}"})
        assert denied.status_code == 403

    _set_admin(email, True)

    with TestClient(app) as client:
        ok = client.get("/dashboard/model-controls", headers={"Authorization": f"Bearer {token}"})
        assert ok.status_code == 200
        data = ok.json()
        assert isinstance(data.get("models"), list)


def test_model_controls_can_disable_model() -> None:
    email = f"controls2-{uuid.uuid4().hex[:8]}@example.com"
    password = "test-password-123"
    _make_user(email, password)
    _set_admin(email, True, is_whitelisted=True)
    token = _create_token(email)

    with TestClient(app) as client:
        get_controls = client.get("/dashboard/model-controls", headers={"Authorization": f"Bearer {token}"})
        assert get_controls.status_code == 200
        models = get_controls.json().get("models", [])
        if not models:
            return

        target = models[0]["id"]
        save = client.post(
            "/dashboard/model-controls",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "models": [
                    {
                        "id": target,
                        "enabled": False,
                        "classification": "nuanced_coding",
                        "effort": "medium",
                    }
                ]
            },
        )
        assert save.status_code == 200

        visible = client.get("/v1/models", headers={"Authorization": f"Bearer {token}"})
        assert visible.status_code == 200
        ids = {m["id"] for m in visible.json().get("data", [])}
        assert target not in ids
