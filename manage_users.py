"""
User management CLI.

Usage:
    python manage_users.py list
    python manage_users.py create <email> <password>
    python manage_users.py change-password <email> <new-password>
    python manage_users.py whitelist <email>
    python manage_users.py promote-admin <email>
    python manage_users.py demote-admin <email>
    python manage_users.py revoke <email>
    python manage_users.py delete <email>
    python manage_users.py token-create <email> [label]
    python manage_users.py token-list <email>
    python manage_users.py token-revoke <email>
"""
import asyncio
import hashlib
import secrets
import sys
from pathlib import Path

# Load .env before importing db
for line in (Path(__file__).parent / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        import os
        k, _, v = line.partition("=")
        if v and "REPLACE_ME" not in v:
            os.environ.setdefault(k.strip(), v.strip())

from middleware.db import GatewayApiToken, User, init_db, SessionLocal
from middleware.auth import _hash
from sqlalchemy import select


async def cmd_list():
    async with SessionLocal() as db:
        users = (await db.execute(select(User).order_by(User.created_at))).scalars().all()
    if not users:
        print("No users.")
        return
    print(f"{'Email':<35} {'Whitelisted':<12} {'Admin':<8} {'ID'}")
    print("-" * 84)
    for u in users:
        print(
            f"{u.email:<35} {'YES' if u.is_whitelisted else 'no':<12} "
            f"{'YES' if u.is_admin else 'no':<8} {u.id}"
        )


async def cmd_create(email: str, password: str):
    async with SessionLocal() as db:
        existing = (await db.execute(select(User).where(User.email == email))).scalars().first()
        if existing:
            print(f"ERROR: {email} already exists")
            return
        user = User(email=email, password_hash=_hash(password), is_whitelisted=False)
        db.add(user)
        await db.commit()
    print(f"Created: {email} (not whitelisted — run `whitelist` to grant access)")


async def cmd_change_password(email: str, password: str):
    async with SessionLocal() as db:
        user = (await db.execute(select(User).where(User.email == email))).scalars().first()
        if not user:
            print(f"ERROR: {email} not found")
            return
        user.password_hash = _hash(password)
        await db.commit()
    print(f"Password updated: {email}")


async def cmd_whitelist(email: str):
    async with SessionLocal() as db:
        user = (await db.execute(select(User).where(User.email == email))).scalars().first()
        if not user:
            print(f"ERROR: {email} not found")
            return
        user.is_whitelisted = True
        await db.commit()
    print(f"Whitelisted: {email}")


async def cmd_revoke(email: str):
    async with SessionLocal() as db:
        user = (await db.execute(select(User).where(User.email == email))).scalars().first()
        if not user:
            print(f"ERROR: {email} not found")
            return
        user.is_whitelisted = False
        await db.commit()
    print(f"Revoked whitelist: {email}")


async def cmd_promote_admin(email: str):
    async with SessionLocal() as db:
        user = (await db.execute(select(User).where(User.email == email))).scalars().first()
        if not user:
            print(f"ERROR: {email} not found")
            return
        user.is_admin = True
        await db.commit()
    print(f"Promoted to admin: {email}")


async def cmd_demote_admin(email: str):
    async with SessionLocal() as db:
        user = (await db.execute(select(User).where(User.email == email))).scalars().first()
        if not user:
            print(f"ERROR: {email} not found")
            return
        user.is_admin = False
        await db.commit()
    print(f"Demoted from admin: {email}")


async def cmd_delete(email: str):
    async with SessionLocal() as db:
        user = (await db.execute(select(User).where(User.email == email))).scalars().first()
        if not user:
            print(f"ERROR: {email} not found")
            return
        await db.delete(user)
        await db.commit()
    print(f"Deleted: {email}")


async def cmd_token_create(email: str, label: str = "default"):
    raw = f"air_{secrets.token_urlsafe(32)}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    async with SessionLocal() as db:
        user = (await db.execute(select(User).where(User.email == email))).scalars().first()
        if not user:
            print(f"ERROR: {email} not found")
            return
        db.add(GatewayApiToken(user_id=user.id, label=label, token_digest=digest))
        await db.commit()
    print(f"Bearer token for {email} (save once; cannot be shown again):\n{raw}")


async def cmd_token_list(email: str):
    async with SessionLocal() as db:
        user = (await db.execute(select(User).where(User.email == email))).scalars().first()
        if not user:
            print(f"ERROR: {email} not found")
            return
        rows = (await db.execute(
            select(GatewayApiToken).where(GatewayApiToken.user_id == user.id)
        )).scalars().all()
    if not rows:
        print("No gateway API tokens.")
        return
    for r in rows:
        print(f"  {r.id}  label={r.label!r}  created={r.created_at}")


async def cmd_token_revoke(email: str):
    async with SessionLocal() as db:
        user = (await db.execute(select(User).where(User.email == email))).scalars().first()
        if not user:
            print(f"ERROR: {email} not found")
            return
        rows = (await db.execute(
            select(GatewayApiToken).where(GatewayApiToken.user_id == user.id)
        )).scalars().all()
        for r in rows:
            await db.delete(r)
        await db.commit()
    print(f"Removed {len(rows)} gateway API token(s) for {email}")


async def main():
    await init_db()
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    cmd = args[0]
    if cmd == "list":
        await cmd_list()
    elif cmd == "create" and len(args) == 3:
        await cmd_create(args[1], args[2])
    elif cmd == "change-password" and len(args) == 3:
        await cmd_change_password(args[1], args[2])
    elif cmd == "whitelist" and len(args) == 2:
        await cmd_whitelist(args[1])
    elif cmd == "promote-admin" and len(args) == 2:
        await cmd_promote_admin(args[1])
    elif cmd == "demote-admin" and len(args) == 2:
        await cmd_demote_admin(args[1])
    elif cmd == "revoke" and len(args) == 2:
        await cmd_revoke(args[1])
    elif cmd == "delete" and len(args) == 2:
        await cmd_delete(args[1])
    elif cmd == "token-create" and len(args) in (2, 3):
        await cmd_token_create(args[1], args[2] if len(args) == 3 else "default")
    elif cmd == "token-list" and len(args) == 2:
        await cmd_token_list(args[1])
    elif cmd == "token-revoke" and len(args) == 2:
        await cmd_token_revoke(args[1])
    else:
        print(__doc__)


asyncio.run(main())
