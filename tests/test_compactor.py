"""Tests for compaction threshold and graceful degradation."""

import pytest
from unittest.mock import AsyncMock, patch
from middleware.compactor import compact, needs_compaction, COMPACTION_THRESHOLD


def _make_history(n: int) -> list[dict]:
    msgs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"message {i}"})
    return msgs


class TestNeedsCompaction:
    def test_below_threshold(self):
        assert not needs_compaction(_make_history(COMPACTION_THRESHOLD - 1))

    def test_at_threshold(self):
        assert needs_compaction(_make_history(COMPACTION_THRESHOLD + 1))

    def test_system_msgs_excluded_from_count(self):
        msgs = [{"role": "system", "content": "be helpful"}] + _make_history(4)
        assert not needs_compaction(msgs)


class TestCompact:
    @pytest.mark.asyncio
    async def test_short_history_passthrough(self):
        msgs = _make_history(3)
        result = await compact(msgs)
        assert result == msgs

    @pytest.mark.asyncio
    async def test_compaction_injects_system_message(self):
        msgs = _make_history(COMPACTION_THRESHOLD + 4)
        mock_resp = AsyncMock()
        mock_resp.choices[0].message.content = "GOAL: test\nSTATE: x=1"

        with patch("middleware.compactor._get_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
            mock_client_fn.return_value = mock_client

            result = await compact(msgs)

        system_msgs = [m for m in result if m["role"] == "system"]
        assert any("COMPACTED CONTEXT" in m["content"] for m in system_msgs)

    @pytest.mark.asyncio
    async def test_groq_failure_degrades_gracefully(self):
        msgs = _make_history(COMPACTION_THRESHOLD + 4)

        with patch("middleware.compactor._get_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("Groq unavailable")
            )
            mock_client_fn.return_value = mock_client

            result = await compact(msgs)

        assert len(result) < len(msgs)
        assert result  # not empty
