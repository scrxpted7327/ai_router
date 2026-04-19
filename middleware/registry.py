"""
Model registry — maps model names to provider + credentials.
Loaded once at startup from environment variables.
"""
from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass
class ModelEntry:
    provider: str       # "anthropic" | "openai" | "gemini" | "openrouter"
    model_id: str       # exact model ID to send to the provider
    api_key: str
    base_url: str | None = None
    extra_headers: dict | None = None


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def build_registry() -> dict[str, ModelEntry]:
    reg: dict[str, ModelEntry] = {}

    # ── Anthropic (Claude) ────────────────────────────────────────────────────
    for alias, model_id in [
        ("claude",              "claude-sonnet-4-6"),
        ("claude-sonnet",       "claude-sonnet-4-6"),
        ("claude-sonnet-4-6",   "claude-sonnet-4-6"),
        ("claude-opus",         "claude-opus-4-7"),
        ("claude-opus-4-7",     "claude-opus-4-7"),
        ("claude-haiku",        "claude-haiku-4-5-20251001"),
    ]:
        reg[alias] = ModelEntry(
            provider="anthropic",
            model_id=model_id,
            api_key=_env("ANTHROPIC_API_KEY"),
        )

    # ── OpenAI / Codex ────────────────────────────────────────────────────────
    for alias, model_id in [
        ("gpt-4o",       "gpt-4o"),
        ("gpt-4o-mini",  "gpt-4o-mini"),
        ("o3",           "o3"),
        ("o3-mini",      "o3-mini"),
        ("codex",        "gpt-4o"),
        ("gpt-5",        "gpt-4o"),   # update when gpt-5 model ID confirmed
    ]:
        reg[alias] = ModelEntry(
            provider="openai",
            model_id=model_id,
            api_key=_env("OPENAI_CODEX_API_KEY"),
            base_url=_env("OPENAI_CODEX_BASE_URL") or None,
        )

    # ── GitHub Copilot ────────────────────────────────────────────────────────
    for alias in ("copilot", "github-copilot"):
        reg[alias] = ModelEntry(
            provider="openai",
            model_id="gpt-4o",
            api_key=_env("GITHUB_COPILOT_TOKEN"),
            base_url="https://api.githubcopilot.com",
            extra_headers={
                "Editor-Version": "vscode/1.90.0",
                "Copilot-Integration-Id": "vscode-chat",
            },
        )

    # ── Gemini ────────────────────────────────────────────────────────────────
    for alias, model_id in [
        ("gemini",             "gemini-2.5-pro"),
        ("gemini-pro",         "gemini-2.5-pro"),
        ("gemini-2.5-pro",     "gemini-2.5-pro"),
        ("gemini-flash",       "gemini-2.0-flash"),
        ("gemini-2.0-flash",   "gemini-2.0-flash"),
    ]:
        reg[alias] = ModelEntry(
            provider="gemini",
            model_id=model_id,
            api_key=_env("GEMINI_API_KEY"),
        )

    # ── Groq (OpenAI-compatible) ──────────────────────────────────────────────
    for alias, model_id in [
        ("groq",                    "llama-3.3-70b-versatile"),
        ("llama",                   "llama-3.3-70b-versatile"),
        ("llama-3.3-70b",           "llama-3.3-70b-versatile"),
        ("groq/llama-3.3-70b",      "llama-3.3-70b-versatile"),
    ]:
        reg[alias] = ModelEntry(
            provider="openai",
            model_id=model_id,
            api_key=_env("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )

    # ── Cerebras (OpenAI-compatible) ──────────────────────────────────────────
    for alias, model_id in [
        ("cerebras",             "llama-3.3-70b"),
        ("cerebras-fast",        "llama-3.3-70b"),
        ("cerebras/llama-3.3",   "llama-3.3-70b"),
    ]:
        reg[alias] = ModelEntry(
            provider="openai",
            model_id=model_id,
            api_key=_env("CEREBRAS_API_KEY"),
            base_url="https://api.cerebras.ai/v1",
        )

    # ── OpenRouter (universal fallback) ───────────────────────────────────────
    for alias, model_id in [
        ("openrouter",            "auto"),
        ("auto",                  "auto"),
        ("free",                  "mistralai/mistral-7b-instruct:free"),
        ("free-mistral",          "mistralai/mistral-7b-instruct:free"),
        ("free-llama",            "meta-llama/llama-3.1-8b-instruct:free"),
        ("deepseek",              "deepseek/deepseek-chat"),
        ("deepseek-r1",           "deepseek/deepseek-r1"),
        ("qwen",                  "qwen/qwen-2.5-72b-instruct"),
    ]:
        reg[alias] = ModelEntry(
            provider="openai",
            model_id=model_id,
            api_key=_env("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            extra_headers={"HTTP-Referer": "http://localhost", "X-Title": "ai-router"},
        )

    # ── ZAI ───────────────────────────────────────────────────────────────────
    reg["zai"] = ModelEntry(
        provider="openai",
        model_id="glm-4-plus",
        api_key=_env("ZAI_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )

    # ── Kilo ──────────────────────────────────────────────────────────────────
    reg["kilo"] = ModelEntry(
        provider="openai",
        model_id="kilo-default",
        api_key=_env("KILO_API_KEY"),
        base_url="https://api.kilo.ai/v1",
    )

    return reg


# Singleton loaded at import time
REGISTRY: dict[str, ModelEntry] = {}


def init(env_path: str | None = None) -> None:
    global REGISTRY
    if env_path:
        _load_dotenv(env_path)
    REGISTRY = build_registry()


def get(model_name: str) -> ModelEntry | None:
    return REGISTRY.get(model_name.lower())


def list_models() -> list[str]:
    return sorted(REGISTRY.keys())


def _load_dotenv(path: str) -> None:
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip()
            if k and v and "REPLACE_ME" not in v:
                os.environ.setdefault(k, v)
