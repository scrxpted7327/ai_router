# AI Router — Session Handoff

## What this project is
FastAPI AI gateway deployed at `https://ai.scrxpted.cc` (DigitalOcean droplet `137.184.61.238`, Cloudflare orange-cloud HTTPS). Routes AI model requests through a unified OpenAI-compatible API with auth, user management, and an Anthropic Messages API proxy so Claude Code works with any model.

## Infrastructure
- **Droplet**: `ssh -i ~/.ssh/do_ai_router root@137.184.61.238`
- **Service**: `systemctl status ai-router` (runs as `airouter` user from `/opt/ai-router`)
- **Nginx**: HTTP on port 80 → `127.0.0.1:4000`
- **Cloudflare**: orange-cloud HTTPS termination (no certbot needed)
- **Repo on server**: `/opt/ai-router` (git pull to deploy)

## Deployment workflow
```bash
# On local machine — push changes
git add -A && git commit -m "..." && git push

# On server
ssh -i ~/.ssh/do_ai_router root@137.184.61.238
cd /opt/ai-router
git pull
systemctl restart ai-router
sleep 8 && curl http://localhost:4000/health
```

Or scp individual files if git is blocked by local server changes:
```bash
scp -i ~/.ssh/do_ai_router middleware/registry.py root@137.184.61.238:/opt/ai-router/middleware/registry.py
```

## Auth system
- Cookie-based sessions (`ai_session` httponly cookie, 7-day TTL)
- Bearer token auth (`Authorization: Bearer air_...`) — stored SHA-256 hashed in DB
- `x-api-key` header auth for `/anthropic/v1/messages` endpoint
- Login rate limit: 10/min per IP, then progressive backoff (failures_past_10 * 60s penalty)
- All `/v1/*` routes require whitelisted user

### User management
```bash
python manage_users.py list
python manage_users.py whitelist <email>
python manage_users.py token-create <email>
python manage_users.py token-list <email>
python manage_users.py token-revoke <token_id>
python manage_users.py change-password <email> <new_password>
```

## Providers currently in registry (middleware/registry.py)
- **Claude via GitHub Copilot**: opus, sonnet, haiku (4.5 series + 3.5 series) — `GITHUB_COPILOT_TOKEN`
- **Copilot GPT-4o**: `github-copilot`, `copilot` aliases — `GITHUB_COPILOT_TOKEN`
- **Groq**: llama-3.3-70b, llama-3.1-8b, deepseek-r1 — `GROQ_API_KEY`
- **Cerebras**: llama-3.3-70b, qwen-3-32b — `CEREBRAS_API_KEY`
- **OpenRouter**: auto, deepseek-v3, deepseek-r1, qwen-2.5, free-mistral, free-llama — `OPENROUTER_API_KEY`
- **ZAI**: glm-4-plus — `ZAI_API_KEY`
- **Kilo**: kilo-default — `KILO_API_KEY`
- **OpenCode**: opencode-default — `OPENCODE_API_KEY` / `OPENCODE_BASE_URL`
- **OpenCode Zen / MiniMax**: MiniMax-Text-01 — `OPENCODE_ZEN_API_KEY` / `OPENCODE_ZEN_BASE_URL`

**Removed** (no longer in registry or .env): Anthropic direct API, Gemini CLI, Google Antigravity, OpenAI Codex

## Anthropic proxy (middleware/anthropic_proxy.py)
Accepts Claude Code / Anthropic SDK requests, routes through the full registry, returns Anthropic-format responses.

Claude Code config:
```
ANTHROPIC_BASE_URL=https://ai.scrxpted.cc/anthropic
ANTHROPIC_API_KEY=air_<your_gateway_token>
```

- `POST /anthropic/v1/messages` — full streaming + non-streaming, any model in registry
- `GET /anthropic/v1/models` — lists all models with rich metadata

## pi_auth.py
Reads `~/.pi/agent/auth.json`, refreshes OAuth tokens, writes to `.env`.
Only writes `GITHUB_COPILOT_TOKEN` now (all other provider mappings removed).

```bash
python pi_auth.py              # refresh + write .env
python pi_auth.py --check      # status only
python pi_auth.py --env-file PATH  # write to specific path
```

## Local changes NOT yet deployed to server
As of end of session:
- `middleware/registry.py` — Gemini/Codex/Antigravity blocks removed
- `pi_auth.py` — `_PROVIDER_MAP` pruned to only `github-copilot`

Deploy with `git push` then `git pull` on server, or scp.

## Pending / unfinished work

### 1. OpenCode routing (user's last request)
User wants to route model requests (Copilot, etc.) through an **opencode instance** rather than calling provider APIs directly. User said "maybe use the tui?" — uncertain about approach.

Options discussed:
1. **OpenCode Zen cloud** — already keyed as `OPENCODE_ZEN_API_KEY`, MiniMax-Text-01 model. Could extend to route Claude models here if Zen supports them.
2. **anomalyco/opencode** — check if `opencode serve` or similar exists for a local API server mode
3. **TUI automation** — fragile, not recommended

`pi` CLI is installed on the droplet (`npm install -g @mariozechner/pi-coding-agent`) and user is logged in. But `pi` has no server/daemon mode — it's a TUI only.

**Next step**: Ask user which direction they want, or check `opencode --help` on the droplet.

### 2. Add more providers from anomalyco/opencode
User requested "add support for all providers" referencing https://github.com/anomalyco/opencode.
Research done but not yet added to registry: Fireworks, xAI/Grok, Mistral, Cohere, Together AI, DeepInfra, Moonshot, Zhipu, Alibaba/Qwen direct, etc.

Add these to `middleware/registry.py` with their respective `_env()` key lookups and `.env.example` entries.

## Key file locations
```
middleware/
  app.py              # FastAPI app, startup, routes
  auth.py             # Cookie + Bearer auth, rate limiting, session routes
  registry.py         # Model registry — all providers
  anthropic_proxy.py  # Anthropic Messages API → OpenAI → back to Anthropic
  db.py               # SQLAlchemy models (User, Session, GatewayApiToken, Conversation)
  providers/
    openai_compat.py  # OpenAI-compatible provider (Copilot, Groq, etc.)
    gemini.py         # Gemini provider (may be unused now)
manage_users.py       # CLI for user/token management
pi_auth.py            # pi-mono OAuth token bridge → .env
.env                  # Live secrets (not in git)
.env.example          # Template
deploy/               # Systemd unit, nginx config, setup scripts
```
