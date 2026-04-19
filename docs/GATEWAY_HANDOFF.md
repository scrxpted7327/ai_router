# AI Gateway — handoff for builders (Claude / contributors)

This document captures how the **ai_router** stack is intended to run in production, how clients authenticate, and how provider credentials flow in. Use it when extending features, debugging deploys, or onboarding.

---

## Production surface (example)

| Item | Value |
|------|--------|
| Public HTTPS hostname | `https://ai.scrxpted.cc` (replace if your domain differs) |
| OpenAI-compatible base URL for clients | `https://ai.scrxpted.cc/v1` |
| Health | `GET https://ai.scrxpted.cc/health` (use **GET**, not HEAD — `/health` may return 405 for HEAD) |
| Interactive API docs (FastAPI default) | `https://ai.scrxpted.cc/docs` |
| Control UI (OpenClaw-style dashboard) | `https://ai.scrxpted.cc/` — static assets in `middleware/static/dashboard/` |

---

## Architecture (single routing layer)

```text
Clients (Cursor, curl, mobile, browser)
  → Cloudflare (DNS / TLS / optional WAF)
  → Nginx on droplet (reverse proxy, SSE-friendly)
  → Uvicorn: FastAPI `middleware.app:app` on 127.0.0.1:4000
  → SQLite `data/gateway.db` (users, sessions, conversations, API token hashes)
  → Outbound HTTP to LLM providers (keys from `/opt/ai-router/.env`)
```

**LiteLLM:** `config.yaml` in this repo is a **reference** for model/provider naming. **Do not** assume a LiteLLM process is required on the 1GB droplet. Production is **one** app: this FastAPI gateway, unless you explicitly add a second proxy for a specific feature.

---

## On the DigitalOcean droplet

| Path / unit | Role |
|-------------|------|
| `/opt/ai-router` | Git checkout of this repo |
| `/opt/ai-router/.env` | Secrets and config (mode `600`, owner `airouter`) |
| `/opt/ai-router/.venv` | Python virtualenv |
| `/opt/ai-router/data/gateway.db` | SQLite database |
| `systemd: ai-router.service` | Runs `uvicorn middleware.app:app --host 127.0.0.1 --port 4000 --workers 1` |
| Nginx | Proxies public 80/443 → app; config from `deploy/nginx-ai.conf` (TLS via Let’s Encrypt when enabled) |
| `deploy/setup_droplet.sh` | Bootstrap: deps, venv, systemd, nginx, ufw, swap |
| `deploy/backup-gateway-db.sh` | Optional scheduled DB backup |
| `deploy/healthcheck.sh` | Optional local health probe |

### Commands you use most

```bash
sudo systemctl status ai-router nginx
curl -sS http://127.0.0.1:4000/health
sudo journalctl -u ai-router -n 100 -f
```

After editing `.env` or deploying code:

```bash
cd /opt/ai-router && sudo -u airouter git pull
sudo -u airouter /opt/ai-router/.venv/bin/pip install -r requirements.txt
sudo systemctl restart ai-router
```

---

## Inbound auth (who may call `/v1/*`)

All **`/v1/*`** routes require a **whitelisted** user via **either**:

1. **`Authorization: Bearer <token>`** — gateway token from `manage_users.py token-create` (prefix `air_...`, stored hashed in DB), **or**
2. **Session cookie** — after `POST /auth/login` with a whitelisted account.

**Registration** (`POST /auth/register`) creates users **not** whitelisted by default. An operator must run **`whitelist`** (or update DB) before `/v1` works for that user.

### Operator CLI (`manage_users.py`)

Run as the same user that owns the DB (typically `airouter` on the server):

```bash
cd /opt/ai-router
sudo -u airouter /opt/ai-router/.venv/bin/python manage_users.py create you@example.com 'password'
sudo -u airouter /opt/ai-router/.venv/bin/python manage_users.py whitelist you@example.com
sudo -u airouter /opt/ai-router/.venv/bin/python manage_users.py token-create you@example.com my-client
```

Save the printed **`air_...`** token once; it cannot be shown again.

Other subcommands: `list`, `revoke`, `delete`, `token-list`, `token-revoke`.

---

## Outbound auth (provider keys to Anthropic, Gemini, Copilot, etc.)

The gateway reads provider keys from **environment** at startup (`middleware/registry.py` loads after `reg.init` in `middleware/app.py`).

**Pi-mono / pi.dev bridge (workstation):**

1. On a machine where **`pi login`** works, tokens land in `~/.pi/agent/auth.json`.
2. Run **`python pi_auth.py`** (or **`./deploy/sync_pi_env.sh`**) to refresh Google-backed entries where implemented and **upsert** mapped keys into `.env`.
3. Copy the resulting secrets to the server (e.g. `scp` to `/opt/ai-router/.env`), fix ownership, restart:

```bash
sudo chown airouter:airouter /opt/ai-router/.env
sudo systemctl restart ai-router
```

Mappings (see `pi_auth.py`): e.g. `google-gemini-cli` → `GEMINI_API_KEY`, `google-antigravity` → `GEMINI_ANTIGRAVITY_KEY`, `anthropic` → `ANTHROPIC_API_KEY`, `github-copilot` → `GITHUB_COPILOT_TOKEN`, `openai-codex` → `OPENAI_CODEX_API_KEY`.

**Refresh reality:** the Python bridge refreshes **Google OAuth** tokens in-script. Other providers may require **`pi login`** again when expired; plan for periodic re-login and re-sync.

Template for server env: `deploy/env.production`. Never commit real `.env`.

---

## Client configuration (Cursor and OpenAI-compatible tools)

| Setting | Value |
|--------|--------|
| Base URL | `https://<your-domain>/v1` |
| API key | Gateway **`air_...`** token (not Anthropic/OpenAI keys — those are server-side only) |
| Model | Any alias exposed by `GET /v1/models` (see `middleware/registry.py` / `middleware/router.py`) |

### curl smoke tests

```bash
curl -sS https://<your-domain>/health
curl -sS https://<your-domain>/v1/models -H "Authorization: Bearer air_YOUR_TOKEN"
curl -sS https://<your-domain>/v1/chat/completions \
  -H "Authorization: Bearer air_YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude","messages":[{"role":"user","content":"Hi"}],"max_tokens":32}'
```

---

## CORS and cookies

- **`CORS_ORIGINS`** in `.env`: comma-separated allowed browser origins, or `*` for development only. For production, set your real HTTPS origins so browser-based tools work.
- **`HTTPS=true`** and a strong **`COOKIE_SECRET`** so session cookies are `Secure` in production.

---

## Memory / conversations (cross-device)

Server-side conversation rows live in SQLite. Authenticated users can use **`/auth/conversations`** (see `middleware/auth.py`) to list/get conversations **after** session login or when the same user is identified via Bearer token — same whitelist rules apply.

---

## TLS / Cloudflare

- Recommended: Cloudflare **Full (strict)** with Let’s Encrypt on the origin; see `deploy/certbot-init.sh` and nginx SSL examples under `deploy/`.
- If Cloudflare is **proxied**, origin firewall and nginx must allow Cloudflare → origin traffic as you’ve configured.

---

## Testing locally / in CI

- `pytest.ini` sets `pythonpath = .` so `pytest` imports `middleware`.
- Run: `pytest tests/`

---

## Security checklist (operators)

- `/opt/ai-router/.env`: `600`, `airouter:airouter`.
- Rotate any API token that was ever pasted into chat or committed.
- Rate-limit `/v1/*` and `/auth/login` at Cloudflare if exposed to the internet.

---

## Repo map (where to edit)

| Area | Location |
|------|-----------|
| HTTP app, CORS, `/v1/*` | `middleware/app.py` |
| Auth, cookies, Bearer, `/auth/*` | `middleware/auth.py` |
| DB models | `middleware/db.py` |
| Model list and provider env wiring | `middleware/registry.py` |
| Routing / aliases | `middleware/router.py` |
| User/token CLI | `manage_users.py` |
| Pi → `.env` bridge | `pi_auth.py`, `deploy/sync_pi_env.sh` |
| Control UI (static) | `middleware/static/dashboard/` (`index.html`, `dashboard.css`, `dashboard.js`) |
| Systemd / nginx / backup scripts | `deploy/` |

---

## Intent for future work

When adding features, preserve: **single FastAPI gateway**, **whitelist + Bearer** for `/v1`, **SQLite** for user/session/memory unless HA requirements change, and **secrets only on the server** — clients use gateway tokens only.
