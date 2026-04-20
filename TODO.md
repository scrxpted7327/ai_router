# TODO — AI Router Code Review

Generated: 2026-04-20

---

## 🔴 CRITICAL — Fix Before Trusting With Sensitive Workloads

### SEC-1: Validate COOKIE_SECRET at startup
**File:** `middleware/auth.py:73`
```python
SECRET = os.getenv("COOKIE_SECRET", "change-me-in-production")
```
The fallback is insecure. If the env var is missing, sessions and encrypted user tokens are trivially forged/decrypted. Add a startup assertion:
```python
if SECRET == "change-me-in-production":
    raise RuntimeError("COOKIE_SECRET env var is not set or is still the default")
```

### SEC-2: Token regeneration silently deletes ALL user tokens
**File:** `middleware/auth.py:267–279`  
`regenerate_token()` deletes every `GatewayApiToken` row for a user when regenerating one. A user with tokens in multiple clients (IDE, CLI, mobile) loses them all. Add a `label` field and only delete+replace the specific token being regenerated; provide a separate `revoke_all_tokens()` path.

### BUG-1: Registry init failure leaves app in broken-but-running state
**File:** `middleware/app.py` lifespan block  
If `reg.init()` fails or returns zero models, the app starts and accepts requests but returns 503 on every routing call with no clear message. Add after init:
```python
if len(reg.list_models()) == 0:
    raise RuntimeError("Registry loaded 0 models — check provider keys in .env")
```

### BUG-2: Hardcoded auto-route model lists not validated against registry
**File:** `middleware/app.py:1136–1162` (approx.)  
`AUTO_LIGHT_ROUTES` and similar dicts contain model IDs like `"qwq-groq"` that may not exist at runtime. If all models in a tier are unavailable, routing silently fails. On startup, log a warning for every hardcoded ID that doesn't resolve in the registry.

---

## 🟡 HIGH — Fix Soon

### SEC-3: No CSRF protection on dashboard mutation endpoints
Dashboard POST/PUT/DELETE endpoints check auth but not CSRF tokens. With `CORS_ORIGINS=*` (the default), a malicious page can POST to these endpoints from a victim's browser using their session cookie. Either set `SameSite=Strict` on session cookies (currently `Lax`) or add a CSRF token header check. At minimum, change the default `CORS_ORIGINS` to not be `*`.

### SEC-4: WebSocket PTY runs a live shell as admin without subprocess safety
**File:** `middleware/app.py:947–975`  
The terminal endpoint spawns `ptyprocess.PtyProcess.spawn(["pi"] + args)` where `args` comes from a query param. Although the route requires admin auth, a compromised admin client could inject shell commands. Use an explicit subprocess arg list and avoid forwarding raw query param tokens to the shell.

### SEC-5: In-memory login rate-limiting resets on restart
**File:** `middleware/auth.py:35`  
`_login_state: dict[str, dict] = {}` is in-process only. Any restart (crash, deploy, systemd reload) resets all brute-force counters. Move to SQLite (a `login_attempts` table with timestamps) or Redis for persistence.

### QUAL-1: Silent `except Exception: pass` throughout codebase
Multiple locations (`anthropic_proxy.py:175`, `app.py` WebSocket read task, `registry.py` model fetch, etc.) swallow exceptions entirely. Replace with at minimum `log.warning("...", exc_info=True)`. Silent failures cause ghost bugs that are impossible to diagnose.

### QUAL-2: `asyncio.create_task()` without error handlers
**File:** `middleware/app.py:1203, 1216` (approx.)  
Fire-and-forget tasks for route analytics and other side effects have no `.add_done_callback()` for exception capture. A crashing task silently disappears. Add:
```python
task = asyncio.create_task(...)
task.add_done_callback(lambda t: t.exception() and log.error(...))
```

### QUAL-3: Compactor silent graceful degradation loses user context without notice
**File:** `middleware/compactor.py:101–104`  
On Groq failure, the compactor silently trims oldest messages instead of summarizing. The user loses context with no indication. Log a warning at minimum; consider surfacing this in the response metadata or as a system message in the conversation.

### QUAL-4: Admin model-control endpoint accepts untyped `dict` payload
**File:** `middleware/app.py` — `POST /dashboard/model-controls`  
Some endpoints take `payload: dict` directly instead of a Pydantic model. Model IDs inserted via these endpoints are not validated against the registry before DB write, leaving orphaned rows. Add a Pydantic schema and validate model ID existence.

---

## 🟢 MEDIUM — Schedule for Next Sprint

### DB-1: Database migrations have no version tracking
**File:** `middleware/db.py` — `init_db()`  
Migrations are inline `ALTER TABLE IF NOT EXISTS` calls with no version number. Out-of-order migration runs or partial failures leave the DB in an inconsistent state with no way to know what ran. Introduce a `schema_version` table or migrate to Alembic.

### DB-2: `provider_model_controls` and `auto_router_configs` can reference non-existent models
**Files:** `middleware/db.py` schema  
No foreign key or application-level validation ensures the `model_id` in these tables actually exists in the registry. Orphan rows cause silent no-ops in routing. Add validation on write and a cleanup task or admin endpoint to prune stale rows.

### SEC-6: No CSP or HSTS headers from application
**File:** `middleware/app.py`  
The dashboard has no `Content-Security-Policy` header, allowing inline scripts and external origins. Add a middleware that sets `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and a minimal CSP. HSTS is provided by Cloudflare but should be added server-side too for defense-in-depth.

### SEC-7: Fernet encryption key tied to COOKIE_SECRET with no rotation path
**File:** `middleware/tokens.py`  
User provider tokens are Fernet-encrypted with a key derived from `COOKIE_SECRET`. If the secret ever needs to rotate, all stored tokens become unrecoverable instantly (no re-encryption path). Add a `TOKEN_ENCRYPTION_KEY` env var separate from the cookie secret, and document a migration procedure.

### OPS-1: `config.yaml` is misleading — it is NOT read by the app
**File:** `config.yaml`  
The file looks like app configuration but is reference documentation only. Someone editing it expecting changes to apply will be confused for a long time. Either remove it, rename it to `config.yaml.reference`, or add a large comment at the top.

### OPS-2: No systemd restart-on-failure or watchdog
**File:** `deploy/` / systemd unit  
The service has no `Restart=on-failure` or `WatchdogSec=` in the unit file. A crash leaves the gateway dark until someone notices. Add `Restart=on-failure` and `RestartSec=5`.

### OPS-3: No automated DB backup
**File:** `deploy/backup-gateway-db.sh` (exists but not wired up)  
The backup script exists but doesn't run automatically. Add a systemd timer or cron job to run it nightly and ship to a remote destination (S3, rsync, etc.).

### TEST-1: No integration tests for full request → response flow
**Files:** `tests/`  
All tests are unit-level. There are no tests that send an actual OpenAI-format request through the stack and verify a proxied response (even against a mock provider). Add at least one happy-path and one error-path integration test per provider backend.

### TEST-2: No tests for error cases (provider down, rate limits, auth failures)
**Files:** `tests/`  
No tests exercise rate-limit fallback, compaction failure, or invalid auth tokens. Add negative-path tests.

---

## 🔵 LOW / NICE-TO-HAVE

### ARCH-1: `registry.py` is ~1,800 lines with too many concerns
Split into:
- `static_registry.py` — hardcoded catalog entries
- `dynamic_loader.py` — models.dev fetch + caching
- `capability_resolver.py` — capability resolution priority chain

### ARCH-2: Routing logic in `app.py::_handle()` is 150+ lines and untestable
Extract task classification, model selection, provider picking, and fallback into a `router_service.py` module that can be unit-tested without spinning up FastAPI.

### ARCH-3: No structured/JSON logging
Current logging is human-readable strings. For production analysis and alerting, add `python-json-logger` or similar so log lines can be parsed by log aggregators (Grafana Loki, CloudWatch, etc.).

### ARCH-4: No per-model user permissions
All whitelisted users can access all enabled models. Add a `user_model_allowlist` table for fine-grained access control (e.g., some users can only use cheap models).

### ARCH-5: Rate-limit 429 responses have no `Retry-After` header
When a provider rate-limits and the gateway returns 429, there's no `Retry-After` hint. Clients (especially Cursor/Claude Code) fall back to defaults. Extract and forward the upstream `retry-after` value.

### FEAT-1: OpenCode integration not implemented
Per `HANDOFF.md`, routing requests through a local OpenCode instance is desired but the approach is unresolved. Decide between TUI automation vs. API surface and create a tracking issue.

### FEAT-2: Additional providers partially wired but not in registry
Fireworks, xAI/Grok, Mistral, Cohere, Together AI, DeepInfra, Moonshot, Zhipu, Alibaba/Qwen are listed in capabilities map but missing from `registry.py` provider functions. Complete or remove stubs.

---

## Summary

| Priority | Count |
|----------|-------|
| 🔴 Critical | 4 |
| 🟡 High | 7 |
| 🟢 Medium | 8 |
| 🔵 Low/Nice-to-have | 7 |
| **Total** | **26** |
