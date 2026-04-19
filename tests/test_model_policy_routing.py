"""Unit tests for model policy routing helpers."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

app_module = importlib.import_module("middleware.app")


def test_policy_pick_prefers_effort_then_preference(monkeypatch) -> None:
    entries = {
        "model-a": SimpleNamespace(model_id="model-a", provider="x"),
        "model-b": SimpleNamespace(model_id="model-b", provider="x"),
        "model-c": SimpleNamespace(model_id="model-c", provider="x"),
    }
    monkeypatch.setattr(app_module.reg, "get", lambda model_id: entries.get(model_id))

    enabled = {"model-a", "model-b", "model-c"}
    meta = {
        "model-a": {"classification": "heavy_reasoning", "effort": "high"},
        "model-b": {"classification": "heavy_reasoning", "effort": "medium"},
        "model-c": {"classification": "heavy_reasoning", "effort": "high"},
    }

    picked = app_module._policy_pick_for_task(
        task_type="heavy_reasoning",
        enabled=enabled,
        meta=meta,
        preferred=["model-b", "model-a"],
    )
    assert picked is not None
    assert picked.model_id == "model-a"


def test_policy_pick_returns_none_without_matching_classification(monkeypatch) -> None:
    monkeypatch.setattr(app_module.reg, "get", lambda model_id: None)
    enabled = {"model-x"}
    meta = {"model-x": {"classification": "multimodal", "effort": "low"}}

    picked = app_module._policy_pick_for_task(
        task_type="code_generation",
        enabled=enabled,
        meta=meta,
        preferred=["model-x"],
    )
    assert picked is None
