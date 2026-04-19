"""Unit tests for task classification and compaction threshold logic."""

import pytest
from middleware.router import TaskType, classify, route


def _msgs(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


class TestClassify:
    def test_heavy_reasoning(self):
        assert classify(_msgs("Analyze the trade-offs of this system design")) == TaskType.HEAVY_REASONING

    def test_code_generation(self):
        assert classify(_msgs("Implement a binary search function in Python")) == TaskType.CODE_GENERATION

    def test_nuanced_coding(self):
        assert classify(_msgs("Refactor this method to improve readability")) == TaskType.NUANCED_CODING

    def test_fast_simple(self):
        assert classify(_msgs("What is the capital of France?")) == TaskType.FAST_SIMPLE

    def test_empty_defaults_to_nuanced(self):
        assert classify([]) == TaskType.NUANCED_CODING


class TestRoute:
    def test_heavy_routes_to_gpt5(self):
        d = route(_msgs("Analyze these architectural trade-offs"))
        assert d.primary == "gpt-5-heavy"
        assert "claude-nuanced" in d.fallbacks

    def test_code_routes_to_codex(self):
        d = route(_msgs("Write a function that implements quicksort"))
        assert d.primary == "gpt-codex"

    def test_nuanced_routes_to_claude(self):
        d = route(_msgs("Refactor this class to use composition over inheritance"))
        assert d.primary == "claude-nuanced"
