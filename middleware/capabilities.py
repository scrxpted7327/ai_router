"""Per-(provider, model) capability resolution.

Resolution order for each flag (first non-null wins):
    1. Admin override stored in `provider_model_controls`
    2. Static per-provider map defined below
    3. `models.dev` capabilities on the model entry
    4. CatalogModel declared flags
    5. Safe default: False

Consumers ask whether a provider can honour a particular request feature
(`reasoning_effort`, thinking/extended output, vision inputs, tool calls).
Relays (openrouter/kilo/opencode) leave most flags as None so we defer to
models.dev — they can only advertise what the upstream provider advertises.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Provider-id -> per-flag overrides. None means "no opinion; consult next source".
_STATIC_PROVIDER_CAPS: dict[str, dict[str, bool | None]] = {
    # --- SDK-native direct-vendor APIs ---
    "openai":             {"effort": True,  "thinking": None, "vision": True,  "tools": True},
    "openai-codex":       {"effort": True,  "thinking": True, "vision": True,  "tools": True},
    "anthropic":          {"effort": None,  "thinking": True, "vision": True,  "tools": True},
    "github-copilot":     {"effort": True,  "thinking": None, "vision": True,  "tools": True},
    "github-copilot-pi":  {"effort": True,  "thinking": True, "vision": True,  "tools": True},
    "gemini":             {"effort": None,  "thinking": True, "vision": True,  "tools": True},
    "google-gemini-cli":  {"effort": None,  "thinking": True, "vision": True,  "tools": True},
    "google-antigravity": {"effort": None,  "thinking": True, "vision": True,  "tools": True},
    "amazon-bedrock":     {"effort": None,  "thinking": True, "vision": True,  "tools": True},
    "deepseek":           {"effort": None,  "thinking": None, "vision": False, "tools": True},
    "xai":                {"effort": True,  "thinking": None, "vision": True,  "tools": True},
    # --- Relays / pass-through aggregators ---
    # Effort/thinking depends entirely on upstream; defer to models.dev.
    "openrouter":         {"effort": None,  "thinking": None, "vision": None,  "tools": True},
    "kilo":               {"effort": None,  "thinking": None, "vision": None,  "tools": True},
    "opencode":           {"effort": None,  "thinking": None, "vision": None,  "tools": True},
    "zai":                {"effort": None,  "thinking": None, "vision": None,  "tools": True},
    # --- Fast inference providers (no effort knob in practice) ---
    "groq":               {"effort": False, "thinking": False, "vision": None,  "tools": True},
    "cerebras":           {"effort": False, "thinking": False, "vision": False, "tools": True},
    "together":           {"effort": False, "thinking": False, "vision": None,  "tools": True},
    "fireworks":          {"effort": False, "thinking": False, "vision": None,  "tools": True},
    "perplexity":         {"effort": None,  "thinking": None,  "vision": False, "tools": False},
    "cohere":             {"effort": False, "thinking": False, "vision": False, "tools": True},
    "mistral":            {"effort": False, "thinking": False, "vision": True,  "tools": True},
}

FLAGS = ("effort", "thinking", "vision", "tools")


@dataclass(frozen=True)
class Capabilities:
    effort: bool = False
    thinking: bool = False
    vision: bool = False
    tools: bool = False

    def supports(self, feature: str) -> bool:
        return bool(getattr(self, feature, False))

    def as_dict(self) -> dict[str, bool]:
        return {flag: getattr(self, flag) for flag in FLAGS}


def _from_models_dev(entry: dict | None) -> dict[str, bool | None]:
    if not isinstance(entry, dict):
        return {flag: None for flag in FLAGS}
    caps = entry.get("capabilities") if isinstance(entry.get("capabilities"), dict) else {}

    def _read(*keys: str) -> bool | None:
        for k in keys:
            if isinstance(caps, dict) and k in caps:
                return bool(caps.get(k))
            if k in entry:
                return bool(entry.get(k))
        return None

    return {
        "effort": _read("reasoning_effort", "effort"),
        "thinking": _read("thinking", "reasoning"),
        "vision": _read("vision", "image_input"),
        "tools": _read("tool_use", "tools", "function_calling"),
    }


def _from_catalog(catalog: Any) -> dict[str, bool | None]:
    if catalog is None:
        return {flag: None for flag in FLAGS}
    return {
        # Catalog `reasoning` boolean doesn't distinguish effort vs thinking, so
        # use it as a weak hint for both — it's outranked by the static map and
        # models.dev when they have opinions.
        "effort": bool(getattr(catalog, "reasoning", False)) or None,
        "thinking": bool(getattr(catalog, "reasoning", False)) or None,
        "vision": bool(getattr(catalog, "vision", False)) or None,
        "tools": None,
    }


def _admin_overrides(row: Any) -> dict[str, bool | None]:
    if row is None:
        return {flag: None for flag in FLAGS}
    return {
        "effort": getattr(row, "supports_effort", None),
        "thinking": getattr(row, "supports_thinking", None),
        "vision": getattr(row, "supports_vision", None),
        "tools": getattr(row, "supports_tools", None),
    }


def provider_model_capabilities(
    provider_id: str,
    model_id: str,
    catalog: Any = None,
    models_dev_entry: dict | None = None,
    admin_override: Any = None,
) -> Capabilities:
    """Resolve capabilities for a provider+model pair.

    `admin_override` is a ProviderModelControl-shaped object (or None).
    `catalog` is a CatalogModel (or None).
    `models_dev_entry` is the raw dict from models.dev for that model (or None).
    """
    sources = (
        _admin_overrides(admin_override),
        _STATIC_PROVIDER_CAPS.get(provider_id, {}),
        _from_models_dev(models_dev_entry),
        _from_catalog(catalog),
    )

    resolved: dict[str, bool] = {}
    for flag in FLAGS:
        value: bool = False
        for src in sources:
            if src.get(flag) is not None:
                value = bool(src[flag])
                break
        resolved[flag] = value

    return Capabilities(**resolved)
