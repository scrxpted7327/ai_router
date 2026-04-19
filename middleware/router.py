"""
Task classifier and model router.

Inspects the last user message and routes to the best available model,
with explicit fallback chains mirroring config.yaml.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class TaskType(str, Enum):
    HEAVY_REASONING = "heavy_reasoning"
    CODE_GENERATION = "code_generation"
    NUANCED_CODING  = "nuanced_coding"
    MULTIMODAL      = "multimodal"
    FAST_SIMPLE     = "fast_simple"


@dataclass
class RouteDecision:
    primary: str
    fallbacks: list[str]
    task_type: TaskType
    reason: str


# ── Keyword sets ─────────────────────────────────────────────────────────────

_HEAVY_REASONING = re.compile(
    r"\b(analyz|evaluat|compar|reason|hypoth|deduc|infer|strateg|architect"
    r"|trade.?off|design pattern|system design|math|proof|theorem)\w*\b",
    re.I,
)
_CODE_GEN = re.compile(
    r"\b(implement|write (a |the )?(function|class|script|program|api|endpoint)"
    r"|generate code|build (a |the )?|create (a |the )?(function|class|module)"
    r"|codex|debug|test suite|unit test|integration test)\w*\b",
    re.I,
)
_NUANCED_CODING = re.compile(
    r"\b(refactor|rewrite|clean up|improve|optimiz|readability|naming|lint"
    r"|code review|smell|pattern|extract method|decompos)\w*\b",
    re.I,
)
_MULTIMODAL = re.compile(
    r"\b(image|screenshot|diagram|chart|vision|multimodal|picture|photo"
    r"|describe (this|the) (image|file)|long.?context|large.?document|pdf)\w*\b",
    re.I,
)
_SIMPLE = re.compile(
    r"\b(what is|define|explain briefly|summarize|list|translate|convert"
    r"|format|parse|lookup|quick)\b",
    re.I,
)


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                return " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            return content
    return ""


def classify(messages: list[dict]) -> TaskType:
    text = _last_user_text(messages)
    if not text:
        return TaskType.NUANCED_CODING

    scores = {
        TaskType.HEAVY_REASONING: len(_HEAVY_REASONING.findall(text)) * 3,
        TaskType.CODE_GENERATION: len(_CODE_GEN.findall(text)) * 2,
        TaskType.NUANCED_CODING:  len(_NUANCED_CODING.findall(text)) * 2,
        TaskType.MULTIMODAL:      len(_MULTIMODAL.findall(text)) * 2,
        TaskType.FAST_SIMPLE:     len(_SIMPLE.findall(text)) * 1,
    }
    winner = max(scores, key=lambda k: scores[k])
    return winner if scores[winner] > 0 else TaskType.NUANCED_CODING


_ROUTES: dict[TaskType, RouteDecision] = {
    TaskType.HEAVY_REASONING: RouteDecision(
        primary="gpt-5-heavy",
        fallbacks=["claude-nuanced", "gemini-pro", "openrouter-gateway"],
        task_type=TaskType.HEAVY_REASONING,
        reason="Complex reasoning — routing to GPT-5.4",
    ),
    TaskType.CODE_GENERATION: RouteDecision(
        primary="gpt-codex",
        fallbacks=["gpt-5-heavy", "claude-nuanced", "opencode-model"],
        task_type=TaskType.CODE_GENERATION,
        reason="Code generation — routing to GPT-5.3-codex",
    ),
    TaskType.NUANCED_CODING: RouteDecision(
        primary="claude-nuanced",
        fallbacks=["gpt-5-heavy", "gemini-pro", "openrouter-gateway"],
        task_type=TaskType.NUANCED_CODING,
        reason="Nuanced coding/refactoring — routing to Claude",
    ),
    TaskType.MULTIMODAL: RouteDecision(
        primary="gemini-pro",
        fallbacks=["gemini-flash", "gpt-5-heavy", "openrouter-gateway"],
        task_type=TaskType.MULTIMODAL,
        reason="Multimodal/long-context — routing to Gemini 2.5 Pro",
    ),
    TaskType.FAST_SIMPLE: RouteDecision(
        primary="cerebras-fast",
        fallbacks=["gemini-flash", "groq-compactor"],
        task_type=TaskType.FAST_SIMPLE,
        reason="Simple query — routing to Cerebras for speed",
    ),
}


def route(messages: list[dict]) -> RouteDecision:
    task = classify(messages)
    return _ROUTES[task]
