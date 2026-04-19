"""
SQLite database — users, sessions, conversation memory.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String, Text, func, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship

DB_PATH = Path(__file__).parent.parent / "data" / "gateway.db"
DB_PATH.parent.mkdir(exist_ok=True)

engine = create_async_engine(f"sqlite+aiosqlite:///{DB_PATH}", echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id           = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email        = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    is_whitelisted = Column(Boolean, default=False, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at   = Column(DateTime(timezone=True), server_default=func.now())

    sessions      = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    gateway_tokens = relationship("GatewayApiToken", back_populates="user", cascade="all, delete-orphan")


class GatewayApiToken(Base):
    """SHA-256 digest of Bearer token for /v1/* (mobile, IDEs, curl without cookies)."""

    __tablename__ = "gateway_api_tokens"

    id           = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id      = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    label        = Column(String, default="default")
    token_digest = Column(String(64), unique=True, nullable=False, index=True)
    created_at   = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="gateway_tokens")


class Session(Base):
    __tablename__ = "sessions"

    id         = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id    = Column(String, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)

    user = relationship("User", back_populates="sessions")


class Conversation(Base):
    __tablename__ = "conversations"

    id              = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id         = Column(String, ForeignKey("users.id"), nullable=False)
    title           = Column(String, default="New conversation")
    model           = Column(String, default="")
    messages        = Column(Text, default="[]")       # JSON array
    compacted_state = Column(Text, default="")          # last Groq summary
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    updated_at      = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="conversations")


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        user_cols = (await conn.execute(text("PRAGMA table_info(users)"))).fetchall()
        user_col_names = {str(row[1]) for row in user_cols}
        if "is_admin" not in user_col_names:
            await conn.execute(text("ALTER TABLE users ADD COLUMN is_admin BOOLEAN NOT NULL DEFAULT 0"))
