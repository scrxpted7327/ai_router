"""Tests for capability-aware provider selection (middleware/provider_selector.py)."""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

reg_module = importlib.import_module("middleware.registry")
sel_module = importlib.import_module("middleware.provider_selector")
caps_module = importlib.import_module("middleware.capabilities")

extract_request_features = sel_module.extract_request_features
pick_provider_for_model  = sel_module.pick_provider_for_model
ProviderOverride         = sel_module.ProviderOverride
RequestFeatures          = sel_module.RequestFeatures
Capabilities             = caps_module.Capabilities


def _make_entry(provider_id, model_id="claude-opus-4-7", caps=None):
    from dataclasses import replace
    entry = reg_module.ModelEntry(
        provider="openai",
        provider_id=provider_id,
        model_id=model_id,
        api_key="test",
        capabilities=caps or Capabilities(),
    )
    return entry


# ── extract_request_features ──────────────────────────────────────────────────

def test_extracts_effort():
    body = {"reasoning_effort": "high", "messages": []}
    f = extract_request_features(body)
    assert f.needs_effort is True
    assert f.needs_thinking is False


def test_extracts_thinking():
    body = {"thinking": {"type": "enabled", "budget_tokens": 8000}}
    f = extract_request_features(body)
    assert f.needs_thinking is True


def test_default_effort_not_flagged():
    body = {"reasoning_effort": "default"}
    f = extract_request_features(body)
    assert f.needs_effort is False


def test_extracts_tools():
    body = {"tools": [{"type": "function", "function": {"name": "f"}}]}
    f = extract_request_features(body)
    assert f.needs_tools is True


def test_extracts_vision():
    body = {"messages": [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:..."}}]}
    ]}
    f = extract_request_features(body)
    assert f.needs_vision is True


# ── pick_provider_for_model ───────────────────────────────────────────────────

def _patch_entries(monkeypatch, entries):
    """Patch registry.get_entries_for_model to return `entries`."""
    monkeypatch.setattr(reg_module, "get_entries_for_model", lambda mid: entries)
    monkeypatch.setattr(reg_module, "get", lambda mid, **kw: entries[0] if entries else None)


def test_pick_prefers_effort_capable(monkeypatch):
    capable  = _make_entry("openai",    caps=Capabilities(effort=True))
    incapable = _make_entry("openrouter", caps=Capabilities(effort=False))
    _patch_entries(monkeypatch, [incapable, capable])

    result = pick_provider_for_model(
        "claude-opus-4-7",
        features=RequestFeatures(needs_effort=True),
    )
    assert result is not None
    assert result.entry.provider_id == "openai"
    assert result.degraded is False


def test_pick_falls_back_when_none_capable(monkeypatch):
    incapable = _make_entry("openrouter", caps=Capabilities(effort=False))
    _patch_entries(monkeypatch, [incapable])

    result = pick_provider_for_model(
        "claude-opus-4-7",
        features=RequestFeatures(needs_effort=True),
    )
    assert result is not None
    assert result.degraded is True


def test_disabled_provider_skipped(monkeypatch):
    capable  = _make_entry("openai",    caps=Capabilities(effort=True))
    disabled = _make_entry("anthropic", caps=Capabilities(effort=True, thinking=True))
    _patch_entries(monkeypatch, [disabled, capable])

    overrides = {("anthropic", "claude-opus-4-7"): ProviderOverride(enabled=False)}
    result = pick_provider_for_model(
        "claude-opus-4-7",
        features=RequestFeatures(needs_effort=True),
        provider_overrides=overrides,
    )
    assert result is not None
    assert result.entry.provider_id == "openai"


def test_returns_none_when_all_disabled(monkeypatch):
    capable = _make_entry("openai", caps=Capabilities(effort=True))
    _patch_entries(monkeypatch, [capable])

    overrides = {("openai", "claude-opus-4-7"): ProviderOverride(enabled=False)}
    result = pick_provider_for_model("claude-opus-4-7", provider_overrides=overrides)
    assert result is None


def test_master_enabled_gate(monkeypatch):
    capable = _make_entry("openai", caps=Capabilities(effort=True))
    _patch_entries(monkeypatch, [capable])

    result = pick_provider_for_model(
        "claude-opus-4-7",
        master_enabled=set(),  # model is disabled at the core level
    )
    assert result is None


def test_priority_tiebreak(monkeypatch):
    a = _make_entry("provider-a", caps=Capabilities(effort=True))
    b = _make_entry("provider-b", caps=Capabilities(effort=True))
    _patch_entries(monkeypatch, [a, b])

    overrides = {
        ("provider-a", "claude-opus-4-7"): ProviderOverride(priority=200),
        ("provider-b", "claude-opus-4-7"): ProviderOverride(priority=50),  # lower = higher priority
    }
    result = pick_provider_for_model(
        "claude-opus-4-7",
        features=RequestFeatures(needs_effort=True),
        provider_overrides=overrides,
    )
    assert result is not None
    assert result.entry.provider_id == "provider-b"
