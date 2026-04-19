"""
Per-user provider token store — Fernet encryption keyed from COOKIE_SECRET.

Usage:
    from .tokens import encrypt_token, decrypt_token, get_user_token

The same COOKIE_SECRET that signs sessions is used to derive the Fernet key,
so no extra env var is needed unless you want independent key rotation.
"""
from __future__ import annotations

import base64
import hashlib
import os

from cryptography.fernet import Fernet
from sqlalchemy import select

from .db import SessionLocal, UserProviderToken


def _fernet() -> Fernet:
    secret = os.environ.get("COOKIE_SECRET", "change-me-in-production")
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
    return Fernet(key)


def encrypt_token(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt_token(ciphertext: str) -> str:
    return _fernet().decrypt(ciphertext.encode()).decode()


async def get_user_token(user_id: str, provider_id: str) -> str | None:
    async with SessionLocal() as db:
        row = (
            await db.execute(
                select(UserProviderToken).where(
                    UserProviderToken.user_id == user_id,
                    UserProviderToken.provider_id == provider_id,
                )
            )
        ).scalars().first()
    if not row:
        return None
    try:
        return decrypt_token(row.encrypted_token)
    except Exception:
        return None


async def set_user_token(user_id: str, provider_id: str, plaintext: str) -> None:
    prefix = plaintext[:8] + "…" if len(plaintext) > 8 else plaintext
    encrypted = encrypt_token(plaintext)
    async with SessionLocal() as db:
        row = (
            await db.execute(
                select(UserProviderToken).where(
                    UserProviderToken.user_id == user_id,
                    UserProviderToken.provider_id == provider_id,
                )
            )
        ).scalars().first()
        if row:
            row.encrypted_token = encrypted
            row.token_prefix = prefix
        else:
            db.add(UserProviderToken(
                user_id=user_id,
                provider_id=provider_id,
                encrypted_token=encrypted,
                token_prefix=prefix,
            ))
        await db.commit()


async def delete_user_token(user_id: str, provider_id: str) -> bool:
    async with SessionLocal() as db:
        row = (
            await db.execute(
                select(UserProviderToken).where(
                    UserProviderToken.user_id == user_id,
                    UserProviderToken.provider_id == provider_id,
                )
            )
        ).scalars().first()
        if not row:
            return False
        await db.delete(row)
        await db.commit()
    return True


async def list_user_tokens(user_id: str) -> list[dict]:
    async with SessionLocal() as db:
        rows = (
            await db.execute(
                select(UserProviderToken).where(UserProviderToken.user_id == user_id)
            )
        ).scalars().all()
    return [
        {
            "provider_id": r.provider_id,
            "token_prefix": r.token_prefix,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        }
        for r in rows
    ]
