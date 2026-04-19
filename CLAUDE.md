# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Start the gateway (port 4000, with hot-reload)
python start.py
# or directly:
uvicorn middleware.app:app --host 0.0.0.0 --port 4000 --reload

# Run all tests
pytest

# Run a single test file
pytest tests/test_router.py

# Run a single test by name
pytest tests/test_router.py::test_classify_heavy_reasoning

# User management
python manage_users.py list
python manage_users.py create <email> <password>
python manage_users.py whitelist <email>
python manage_users.py promote-admin <email>
python manage_users.py token-create <email>   # creates Bearer token for /v1/*

# pi-mono OAuth token refresh (GitHub Copilot, Anthropic, Gemini, etc.)
python pi_auth.py --login    # interactive login
python pi_auth.py --check    # check token expiry without refreshing
```

## Architecture

This is a **self-hosted AI gateway** — an OpenAI-compatible FastAPI server that proxies requests to multiple AI providers behind a single endpoint with auth, routing, and context compaction.

### Request flow

1. **Auth** (`middleware/auth.py`) — validates httponly session cookie or `Authorization: Bearer air_...` token against SQLite. Users must be whitelisted; admin endpoints require `is_admin=True`.
2. **Format normalization** (`middleware/format_adapter.py`) — Cursor's Responses API format (`/v1/responses`) is translated to Chat Completions before processing; the SSE stream is re-wrapped on the way out.
3. **Compaction** (`middleware/compactor.py`) — if the conversation exceeds `COMPACTION_THRESHOLD` (default 8) non-system messages, older turns are summarized via Groq (`llama-3.3-70b-versatile`) into a structured truth-state block, keeping `PRESERVE_TAIL` (default 3) recent messages verbatim.
4. **Routing** (`middleware/router.py`) — if no explicit model is given, the last user message is classified (heavy_reasoning / code_generation / nuanced_coding / multimodal / fast_simple) via regex scoring and mapped to a preferred provider + fallback chain.
5. **Registry lookup** (`middleware/registry.py`) — resolves model name or alias → `ModelEntry` (provider, api_key, base_url, extra_headers). Registry is built at startup from env vars; providers are skipped if their keys are missing. Model metadata is enriched from models.dev (1h cache at `~/.cache/ai_router/models.dev.json`).
6. **Admin policy** (`ModelControl` table) — per-model enable/disable, classification, and effort overrides take precedence; disabled models return 403.
7. **Provider dispatch** (`middleware/providers/`) — three provider backends:
   - `openai_compat.py` — used for OpenAI, Groq, Cerebras, Copilot, OpenRouter, ZAI, Kilo, OpenCode (all OpenAI-format APIs)
   - `gemini.py` — Google Gemini via `google-genai`
   - `bedrock.py` — AWS Bedrock via `boto3`

### Anthropic proxy (`/anthropic/*`)

`middleware/anthropic_proxy.py` exposes a native **Anthropic Messages API** surface (`/anthropic/v1/messages`, `/anthropic/v1/models`) so Claude Code / the `claude` CLI can use the gateway without modification. Auth uses `x-api-key` instead of a cookie. Requests are translated Anthropic→OpenAI internally, responses translated back.

Configure Claude Code to use it:
```
ANTHROPIC_BASE_URL=https://<host>/anthropic
ANTHROPIC_API_KEY=air_...
```

### Database

SQLite at `data/gateway.db` (auto-created). Schema: `users`, `sessions`, `gateway_api_tokens`, `model_controls`, `conversations`. `init_db()` runs migrations inline (ALTER TABLE for columns added after initial deploy).

### Provider registry (`middleware/registry.py`)

Each provider is a `_*_provider()` function that returns `(ProviderConfig, tuple[CatalogModel, ...])` or `None` if its env key is absent. `build_registry()` aggregates them, applies `MODEL_ENABLED_PROVIDERS` / `MODEL_DISABLED_PROVIDERS` / `MODEL_WHITELIST` / `MODEL_BLACKLIST` env filters, and builds a flat alias index so any of `model_id`, `provider/model_id`, `api/model_id`, or declared aliases resolve to the same `ModelEntry`.

### Key env vars

Copy `.env.example` → `.env`. Key vars:
- Per-provider keys: `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `GITHUB_COPILOT_TOKEN`, `GEMINI_API_KEY`, etc.
- `AWS_REGION` + one of `AWS_ACCESS_KEY_ID`, `AWS_PROFILE`, or `AWS_BEARER_TOKEN_BEDROCK` — enables Bedrock
- `COMPACTION_THRESHOLD`, `PRESERVE_TAIL` — tune context compaction
- `MODEL_ENABLED_PROVIDERS`, `MODEL_DISABLED_PROVIDERS` — comma-separated provider filter
- `CORS_ORIGINS` — comma-separated origins (defaults to `*`)

### Dashboard

Static SPA served at `/` (`middleware/static/dashboard/`). Admin-only WebSocket terminal at `/terminal?cmd=login` runs `pi login` for OAuth token refresh.
