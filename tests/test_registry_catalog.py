"""Tests for canonical registry catalog behavior."""

from __future__ import annotations

import asyncio
import importlib
import json
from types import SimpleNamespace


registry = importlib.import_module("middleware.registry")
app_module = importlib.import_module("middleware.app")


def test_registry_lists_canonical_models_once_and_resolves_aliases(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN", "cop-token")
    monkeypatch.delenv("MODEL_ENABLED_PROVIDERS", raising=False)
    monkeypatch.delenv("MODEL_DISABLED_PROVIDERS", raising=False)
    monkeypatch.delenv("MODEL_WHITELIST", raising=False)
    monkeypatch.delenv("MODEL_BLACKLIST", raising=False)
    monkeypatch.setattr(registry, "_load_models_dev", lambda: {})

    state = registry.build_registry()
    ids = list(state.by_canonical_id.keys())

    assert ids.count("claude-sonnet-4-5") == 1
    assert state.aliases["claude"] == "claude-sonnet-4-5"
    assert state.aliases["github-copilot"] == "gpt-4o"


def test_registry_can_filter_by_provider(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN", "cop-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-token")
    monkeypatch.setenv("MODEL_ENABLED_PROVIDERS", "openrouter")
    monkeypatch.setattr(registry, "_load_models_dev", lambda: {})

    state = registry.build_registry()

    assert state.by_canonical_id
    assert all(entry.provider_id == "openrouter" for entry in state.by_canonical_id.values())


def test_bedrock_models_register_without_api_key(monkeypatch) -> None:
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_PROFILE", "default")
    monkeypatch.setenv("BEDROCK_MODELS", "anthropic.claude-3-5-sonnet-20241022-v2:0")
    monkeypatch.setattr(registry, "_load_models_dev", lambda: {})

    state = registry.build_registry()
    entry = state.by_canonical_id["us.anthropic.claude-3-5-sonnet-20241022-v2:0"]

    assert entry.provider == "bedrock"
    assert entry.provider_id == "amazon-bedrock"
    assert entry.api_key == ""
    assert entry.options["region"] == "us-east-1"


def test_handle_allows_bedrock_without_api_key(monkeypatch) -> None:
    entry = SimpleNamespace(
        provider="bedrock",
        provider_id="amazon-bedrock",
        model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        api_key="",
        options={"region": "us-east-1"},
    )
    monkeypatch.setattr(app_module, "_model_control_index", lambda: asyncio.sleep(0, result=({entry.model_id}, {})))
    monkeypatch.setattr(app_module.reg, "get", lambda model_id: entry)

    async def _fake_complete(_entry, _body):
        return {"ok": True}

    monkeypatch.setattr(app_module, "_complete", _fake_complete)

    response = asyncio.run(
        app_module._handle(
            {"model": entry.model_id, "messages": [], "stream": False},
            is_responses_api=False,
            user=None,
        )
    )

    assert json.loads(response.body.decode("utf-8")) == {"ok": True}
