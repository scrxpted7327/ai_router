"""Tests for the capability resolver (middleware/capabilities.py)."""
from __future__ import annotations

import importlib
import sys

import pytest

caps_module = importlib.import_module("middleware.capabilities")
Capabilities = caps_module.Capabilities
provider_model_capabilities = caps_module.provider_model_capabilities


class _FakeCatalog:
    def __init__(self, reasoning=False, vision=False):
        self.reasoning = reasoning
        self.vision = vision


class _FakePMC:
    """Minimal ProviderModelControl-shaped object for overrides."""
    def __init__(self, **kwargs):
        self.supports_effort   = kwargs.get("supports_effort")
        self.supports_thinking = kwargs.get("supports_thinking")
        self.supports_vision   = kwargs.get("supports_vision")
        self.supports_tools    = kwargs.get("supports_tools")


# ── Static provider map ───────────────────────────────────────────────────────

def test_openai_effort_true():
    caps = provider_model_capabilities("openai", "gpt-4o")
    assert caps.effort is True


def test_groq_effort_false():
    caps = provider_model_capabilities("groq", "llama-3.3-70b")
    assert caps.effort is False
    assert caps.thinking is False


def test_anthropic_thinking_true():
    caps = provider_model_capabilities("anthropic", "claude-opus-4-7")
    assert caps.thinking is True


def test_gemini_thinking_true():
    caps = provider_model_capabilities("gemini", "gemini-2.5-pro")
    assert caps.thinking is True


def test_relay_defers_effort_to_models_dev_not_static():
    """openrouter has no opinion — should stay False when no models.dev data."""
    caps = provider_model_capabilities("openrouter", "claude-opus-4-7")
    assert caps.effort is False  # no models.dev, no catalog hint → default False


def test_relay_picks_up_models_dev_thinking():
    """When models.dev says the model supports reasoning, relay inherits thinking=True."""
    caps = provider_model_capabilities(
        "openrouter",
        "claude-opus-4-7",
        models_dev_entry={"capabilities": {"reasoning": True}},
    )
    assert caps.thinking is True


def test_relay_picks_up_models_dev_effort_via_effort_key():
    """When models.dev has a 'reasoning_effort' capability key, relay inherits effort=True."""
    caps = provider_model_capabilities(
        "openrouter",
        "some-model",
        models_dev_entry={"capabilities": {"reasoning_effort": True}},
    )
    assert caps.effort is True


def test_catalog_vision_hint_used_by_relay():
    catalog = _FakeCatalog(vision=True)
    caps = provider_model_capabilities("openrouter", "some-model", catalog=catalog)
    assert caps.vision is True


# ── Admin override wins ───────────────────────────────────────────────────────

def test_admin_override_wins_over_static():
    """Admin can force-disable effort even for openai."""
    override = _FakePMC(supports_effort=False)
    caps = provider_model_capabilities("openai", "o4-mini", admin_override=override)
    assert caps.effort is False


def test_admin_override_enables_effort_for_groq():
    """Admin can force-enable effort even for groq (e.g. they added a beta key)."""
    override = _FakePMC(supports_effort=True)
    caps = provider_model_capabilities("groq", "llama-3.3-70b", admin_override=override)
    assert caps.effort is True


def test_admin_override_null_falls_through_to_static():
    """None override means 'use next source', not False."""
    override = _FakePMC(supports_effort=None)
    caps = provider_model_capabilities("openai", "gpt-4o", admin_override=override)
    assert caps.effort is True  # static map says openai supports effort


# ── Unknown provider defaults ─────────────────────────────────────────────────

def test_unknown_provider_defaults_all_false():
    caps = provider_model_capabilities("totally-unknown-provider", "some-model")
    assert caps.effort is False
    assert caps.thinking is False
    assert caps.vision is False
    assert caps.tools is False


def test_supports_method():
    caps = Capabilities(effort=True, thinking=False, vision=True, tools=True)
    assert caps.supports("effort")
    assert not caps.supports("thinking")
    assert caps.supports("vision")
    assert not caps.supports("nonexistent_flag")
