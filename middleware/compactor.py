"""
Groq-backed compaction engine.

Condenses long message histories into a structured "truth state" —
active goals, key decisions, and current variables — before forwarding
to heavyweight models.
"""

import asyncio
import os
from typing import Any

from groq import AsyncGroq

COMPACTION_THRESHOLD = int(os.getenv("COMPACTION_THRESHOLD", "8"))  # messages
PRESERVE_TAIL = int(os.getenv("PRESERVE_TAIL", "3"))                 # recent msgs kept verbatim

_COMPACTION_PROMPT = """\
You are a context compactor. Given a conversation history, extract the minimal \
"truth state" needed to continue working accurately. Output ONLY a structured \
summary using this exact format:

GOAL: <one-sentence description of the active task>
DECISIONS: <bullet list of key choices already made>
STATE: <relevant variables, values, filenames, constraints>
PENDING: <next actions not yet completed>
CONTEXT: <any crucial background facts>

Be terse. Omit anything recoverable from standard knowledge. \
Never include pleasantries or filler.
"""

_groq_client: AsyncGroq | None = None


def _get_client() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client


def _format_history_for_compaction(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        if isinstance(content, list):
            # Handle multipart content — extract text blocks only
            content = " ".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
        lines.append(f"[{role}]: {content[:1200]}")  # cap per-message length
    return "\n\n".join(lines)


async def compact(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Returns a compacted message list:
      [system: compacted truth state] + last PRESERVE_TAIL messages

    Falls back to the original list if Groq is unavailable.
    """
    if len(messages) <= COMPACTION_THRESHOLD:
        return messages

    # Separate system prompt (if any) — we keep it intact
    system_msgs = [m for m in messages if m.get("role") == "system"]
    history = [m for m in messages if m.get("role") != "system"]

    to_compact = history[:-PRESERVE_TAIL] if len(history) > PRESERVE_TAIL else history
    tail = history[-PRESERVE_TAIL:] if len(history) > PRESERVE_TAIL else []

    try:
        client = _get_client()
        resp = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _COMPACTION_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Compact this conversation:\n\n"
                        + _format_history_for_compaction(to_compact)
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=512,
        )
        summary = resp.choices[0].message.content.strip()
        compacted_system = {
            "role": "system",
            "content": (
                "[COMPACTED CONTEXT — do not re-summarize]\n"
                + summary
            ),
        }
        return system_msgs + [compacted_system] + tail

    except Exception as exc:  # network error, rate limit, etc.
        # Graceful degradation: trim oldest non-system messages instead
        trimmed = history[len(to_compact) :]
        return system_msgs + trimmed


def needs_compaction(messages: list[dict]) -> bool:
    non_system = [m for m in messages if m.get("role") != "system"]
    return len(non_system) > COMPACTION_THRESHOLD
