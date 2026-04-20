"""Capability-aware provider selection.

Given a canonical model_id and a normalized OpenAI-style request body, pick
the best serving provider. The picker is pure — all DB-loaded state
(`ProviderModelControl` rows, disabled providers, master enable set) is
passed in.

Scoring prefers providers that honour every required feature of the
request. If none can, return the top partial match and set
`degraded=True` so the caller can log it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from . import registry as reg
from .capabilities import Capabilities


@dataclass(frozen=True)
class RequestFeatures:
    needs_effort: bool = False
    needs_thinking: bool = False
    needs_vision: bool = False
    needs_tools: bool = False

    @property
    def required(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.needs_effort: out.append("effort")
        if self.needs_thinking: out.append("thinking")
        if self.needs_vision: out.append("vision")
        if self.needs_tools: out.append("tools")
        return tuple(out)


@dataclass(frozen=True)
class ProviderOverride:
    """Per-(provider, model) admin overrides loaded from ProviderModelControl."""
    enabled: bool = True
    priority: int = 100
    capabilities: Capabilities | None = None  # if present, overrides registry capabilities


@dataclass(frozen=True)
class Selection:
    entry: reg.ModelEntry
    degraded: bool
    reason: str


def _has_image_parts(messages: list) -> bool:
    for msg in messages or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    t = part.get("type", "")
                    if t in ("image_url", "input_image", "image"):
                        return True
    return False


def extract_request_features(body: dict) -> RequestFeatures:
    has_effort = False
    has_thinking = False
    for key in ("reasoning_effort", "reasoning"):
        if body.get(key) not in (None, "", "default"):
            has_effort = True
            break
    if body.get("thinking") not in (None, False, {}):
        has_thinking = True
    tools = body.get("tools")
    has_tools = isinstance(tools, list) and len(tools) > 0
    has_vision = _has_image_parts(body.get("messages") or [])
    return RequestFeatures(
        needs_effort=has_effort,
        needs_thinking=has_thinking,
        needs_vision=has_vision,
        needs_tools=has_tools,
    )


def _score_entry(
    entry: reg.ModelEntry,
    caps: Capabilities,
    priority: int,
    order_index: int,
    features: RequestFeatures,
) -> tuple:
    # Higher = better. Sort descending on the full tuple.
    matched = sum(
        1
        for flag, need in (
            ("effort", features.needs_effort),
            ("thinking", features.needs_thinking),
            ("vision", features.needs_vision),
            ("tools", features.needs_tools),
        )
        if need and caps.supports(flag)
    )
    # Negate priority so a lower numeric priority sorts higher.
    return (matched, -priority, -order_index)


def pick_provider_for_model(
    model_id: str,
    *,
    body: dict | None = None,
    features: RequestFeatures | None = None,
    master_enabled: set[str] | None = None,
    provider_overrides: dict[tuple[str, str], ProviderOverride] | None = None,
) -> Selection | None:
    """Select the best ModelEntry for `model_id`.

    `master_enabled` is the set of canonical model_ids passing core-model
    enable checks. If None, no core-model filter is applied.
    `provider_overrides` is keyed by (provider_id, canonical_model_id).
    """
    if features is None:
        features = extract_request_features(body or {})

    # Resolve alias → canonical id.
    entries = reg.get_entries_for_model(model_id)
    if not entries:
        # Fall back to a direct lookup: maybe model_id itself resolves via alias.
        anchor = reg.get(model_id)
        if anchor is None:
            return None
        entries = reg.get_entries_for_model(anchor.model_id) or [anchor]

    canonical = entries[0].model_id
    if master_enabled is not None and canonical not in master_enabled:
        return None

    overrides = provider_overrides or {}
    candidates: list[tuple[tuple, reg.ModelEntry, Capabilities]] = []
    for idx, entry in enumerate(entries):
        override = overrides.get((entry.provider_id, canonical))
        if override is not None and not override.enabled:
            continue
        caps = (override.capabilities if override and override.capabilities is not None else entry.capabilities)
        priority = override.priority if override else 100
        score = _score_entry(entry, caps, priority, idx, features)
        candidates.append((score, entry, caps))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_entry, best_caps = candidates[0]

    required = features.required
    degraded = False
    reason = "match"
    if required:
        matches_all = all(best_caps.supports(flag) for flag in required)
        if not matches_all:
            degraded = True
            missing = [f for f in required if not best_caps.supports(f)]
            reason = f"degraded: missing {','.join(missing)}"
    return Selection(entry=best_entry, degraded=degraded, reason=reason)
