"""
Cookie-based auth with whitelist enforcement, plus optional Bearer tokens.

Flow:
  POST /auth/register  → create account (not whitelisted by default)
  POST /auth/login     → set httponly session cookie
  POST /auth/logout    → clear cookie
  GET  /auth/me        → current user info

All /v1/* routes require a valid whitelisted session OR Authorization: Bearer <token>
(see `python manage_users.py token-create <email>`).
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response
import bcrypt
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .db import Conversation, GatewayApiToken, Session, SessionLocal, User

COOKIE_NAME   = "ai_session"
SESSION_DAYS  = int(os.getenv("SESSION_DAYS", "7"))
SECRET        = os.getenv("COOKIE_SECRET", "change-me-in-production")

def _hash(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def _verify(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

router = APIRouter(prefix="/auth", tags=["auth"])


# ── DB session dependency ─────────────────────────────────────────────────────

async def get_db():
    async with SessionLocal() as db:
        yield db


# ── Pydantic models ───────────────────────────────────────────────────────────

class Credentials(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    id: str
    email: str
    is_whitelisted: bool


# ── Cookie helpers ────────────────────────────────────────────────────────────

def _set_cookie(response: Response, session_id: str) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=session_id,
        httponly=True,
        secure=os.getenv("HTTPS", "false").lower() == "true",
        samesite="lax",
        max_age=SESSION_DAYS * 86400,
        path="/",
    )


def _clear_cookie(response: Response) -> None:
    response.delete_cookie(COOKIE_NAME, path="/")


# ── Session resolution (used by API middleware) ───────────────────────────────

def _bearer_digest(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def resolve_session(
    request: Request,
    session_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
    db: AsyncSession = Depends(get_db),
) -> User:
    auth = request.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        raw = auth[7:].strip()
        if raw:
            digest = _bearer_digest(raw)
            row = await db.execute(
                select(GatewayApiToken).where(GatewayApiToken.token_digest == digest)
            )
            tok = row.scalars().first()
            if tok:
                user_row = await db.get(User, tok.user_id)
                if user_row:
                    return user_row
        raise HTTPException(status_code=401, detail="Invalid or unknown API token")

    if not session_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    now = datetime.now(timezone.utc)
    row = await db.execute(
        select(Session).where(Session.id == session_id, Session.expires_at > now)
    )
    sess = row.scalars().first()
    if not sess:
        raise HTTPException(status_code=401, detail="Session expired or invalid")

    user_row = await db.get(User, sess.user_id)
    if not user_row:
        raise HTTPException(status_code=401, detail="User not found")
    return user_row


async def require_whitelisted(user: User = Depends(resolve_session)) -> User:
    if not user.is_whitelisted:
        raise HTTPException(
            status_code=403,
            detail="Access denied. Your account is not whitelisted for AI model access.",
        )
    return user


# ── Auth routes ───────────────────────────────────────────────────────────────

@router.post("/register", response_model=UserOut)
async def register(creds: Credentials, db: AsyncSession = Depends(get_db)):
    existing = (await db.execute(select(User).where(User.email == creds.email))).scalars().first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=creds.email,
        password_hash=_hash(creds.password),
        is_whitelisted=False,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return UserOut(id=user.id, email=user.email, is_whitelisted=user.is_whitelisted)


@router.post("/login")
async def login(creds: Credentials, response: Response, db: AsyncSession = Depends(get_db)):
    user = (await db.execute(select(User).where(User.email == creds.email))).scalars().first()
    if not user or not _verify(creds.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    expires = datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS)
    sess = Session(user_id=user.id, expires_at=expires)
    db.add(sess)
    await db.commit()

    _set_cookie(response, sess.id)
    return {"message": "Logged in", "whitelisted": user.is_whitelisted}


@router.post("/logout")
async def logout(
    response: Response,
    session_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
    db: AsyncSession = Depends(get_db),
):
    if session_id:
        sess = await db.get(Session, session_id)
        if sess:
            await db.delete(sess)
            await db.commit()
    _clear_cookie(response)
    return {"message": "Logged out"}


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(resolve_session)):
    return UserOut(id=user.id, email=user.email, is_whitelisted=user.is_whitelisted)


# ── Conversation history routes ───────────────────────────────────────────────

@router.get("/conversations")
async def list_conversations(
    user: User = Depends(require_whitelisted),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
        .limit(50)
    )).scalars().all()
    return [{"id": r.id, "title": r.title, "model": r.model, "updated_at": r.updated_at} for r in rows]


@router.get("/conversations/{conv_id}")
async def get_conversation(
    conv_id: str,
    user: User = Depends(require_whitelisted),
    db: AsyncSession = Depends(get_db),
):
    row = await db.get(Conversation, conv_id)
    if not row or row.user_id != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    return {
        "id": row.id, "title": row.title, "model": row.model,
        "messages": json.loads(row.messages),
        "compacted_state": row.compacted_state,
    }
