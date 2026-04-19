"""
pi-mono OAuth token bridge.

Reads ~/.pi/agent/auth.json, refreshes expired access tokens where possible,
and writes live tokens into .env for LiteLLM consumption.

Usage:
    python pi_auth.py                      # refresh + write ./.env
    python pi_auth.py --env-file PATH       # write merged keys to PATH (e.g. before scp to droplet)
    python pi_auth.py --check               # status only, no writes
    python pi_auth.py --force               # refresh all tokens even if still valid

Server workflow: run `pi login` on a trusted machine, then `python pi_auth.py --env-file .env`,
scp `.env` to `/opt/ai-router/.env` on the droplet, `chown airouter:airouter` + `systemctl restart ai-router`.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── auth.json locations (pi-mono uses the first one found) ───────────────────
_AUTH_SEARCH_PATHS: list[Path] = [
    Path.home() / ".pi" / "agent" / "auth.json",
    Path.home() / ".pi" / "auth.json",
    Path.home() / ".config" / "pi" / "auth.json",
    Path.cwd() / "auth.json",
    Path.cwd() / ".pi" / "auth.json",
]

_DEFAULT_ENV = Path(__file__).parent / ".env"

# ── Provider → env var mapping ────────────────────────────────────────────────
# Matches actual keys in pi-mono auth.json
_PROVIDER_MAP: dict[str, dict[str, str]] = {
    "anthropic":         {"env": "ANTHROPIC_API_KEY",    "name": "Claude (Anthropic)"},
    "github-copilot":    {"env": "GITHUB_COPILOT_TOKEN", "name": "GitHub Copilot"},
    "google-gemini-cli": {"env": "GEMINI_API_KEY",        "name": "Gemini CLI"},
    "google-antigravity":{"env": "GEMINI_ANTIGRAVITY_KEY","name": "Gemini Antigravity"},
    "openai-codex":      {"env": "OPENAI_CODEX_API_KEY", "name": "Codex (OpenAI)"},
}

# ── Google OAuth2 client credentials (from pi-mono source) ───────────────────
# These are public OAuth client IDs used by the Gemini CLI / gcloud tools
_GOOGLE_CLIENT_ID     = "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com"
_GOOGLE_CLIENT_SECRET = "d-FL95Q19q7MQmFpd7hHD0Ty"
_GOOGLE_TOKEN_URL     = "https://oauth2.googleapis.com/token"


# ── Token helpers ─────────────────────────────────────────────────────────────

def _now() -> float:
    return time.time()


def _is_expired(entry: dict) -> bool:
    exp = entry.get("expires", 0)
    if exp > 1e12:
        exp /= 1000
    return exp > 0 and _now() > exp - 60   # 60s buffer


def _access_token(entry: dict) -> str:
    return entry.get("access") or entry.get("accessToken") or ""


def _refresh_token(entry: dict) -> str:
    return entry.get("refresh") or entry.get("refreshToken") or ""


def _exp_str(entry: dict) -> str:
    exp = entry.get("expires", 0)
    if exp > 1e12:
        exp /= 1000
    if not exp:
        return "no expiry"
    return datetime.fromtimestamp(exp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ── Refresh implementations ───────────────────────────────────────────────────

def _refresh_google(entry: dict) -> dict | None:
    """Standard Google OAuth2 refresh — works for gemini-cli and antigravity."""
    rt = _refresh_token(entry)
    if not rt:
        return None
    data = urllib.parse.urlencode({
        "client_id":     _GOOGLE_CLIENT_ID,
        "client_secret": _GOOGLE_CLIENT_SECRET,
        "refresh_token": rt,
        "grant_type":    "refresh_token",
    }).encode()
    try:
        req = urllib.request.Request(_GOOGLE_TOKEN_URL, data=data,
                                     headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
        return {
            **entry,
            "access":  body["access_token"],
            "expires": (_now() + body.get("expires_in", 3600)) * 1000,
        }
    except Exception as exc:
        print(f"    Google refresh failed: {exc}")
        return None


def _refresh_entry(provider: str, entry: dict) -> dict:
    """Attempt to refresh. Returns updated entry (or original on failure)."""
    if provider in ("google-gemini-cli", "google-antigravity"):
        updated = _refresh_google(entry)
        if updated:
            print(f"    Refreshed via Google OAuth2")
            return updated
    # Anthropic, GitHub Copilot, OpenAI Codex require pi CLI or browser flow
    print(f"    Cannot auto-refresh {provider} — run `pi login` in a terminal")
    return entry


# ── Load + refresh auth.json ──────────────────────────────────────────────────

def find_auth_file() -> Path | None:
    for p in _AUTH_SEARCH_PATHS:
        if p.exists():
            return p
    return None


def load_and_refresh(auth_path: Path, force: bool = False) -> dict[str, dict]:
    """
    Returns { provider_key: entry } with access tokens refreshed where possible.
    Also writes refreshed tokens back to auth.json.
    """
    raw: dict = json.loads(auth_path.read_text(encoding="utf-8"))
    dirty = False

    for provider, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        if force or _is_expired(entry):
            print(f"  Refreshing {provider} ...")
            updated = _refresh_entry(provider, entry)
            if updated is not entry:
                raw[provider] = updated
                dirty = True

    if dirty:
        auth_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        print(f"  Saved refreshed tokens -> {auth_path}\n")

    return raw


# ── Write to .env ─────────────────────────────────────────────────────────────

def _upsert(lines: list[str], key: str, value: str) -> list[str]:
    result, done = [], False
    for line in lines:
        if line.startswith(f"{key}="):
            if not done:
                result.append(f"{key}={value}")
                done = True
        else:
            result.append(line)
    if not done:
        result.append(f"{key}={value}")
    return result


def write_to_env(providers: dict[str, dict], env_path: Path) -> None:
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []

    written, skipped = [], []
    for provider, entry in providers.items():
        mapping = _PROVIDER_MAP.get(provider)
        if not mapping:
            continue
        token = _access_token(entry)
        if not token:
            skipped.append(f"  SKIP  {mapping['name']}: no access token")
            continue
        if _is_expired(entry):
            skipped.append(f"  SKIP  {mapping['name']}: still expired (needs `pi login`)")
            continue
        lines = _upsert(lines, mapping["env"], token)
        written.append(f"  WROTE {mapping['name']} -> {mapping['env']}")

    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for msg in written + skipped:
        print(msg)

    if written:
        print(f"\nUpdated: {env_path}")


# ── Status printer ────────────────────────────────────────────────────────────

def print_status(providers: dict[str, dict]) -> None:
    for provider, entry in providers.items():
        if not isinstance(entry, dict):
            continue
        mapping = _PROVIDER_MAP.get(provider)
        name    = mapping["name"] if mapping else provider
        env_var = mapping["env"]  if mapping else "—"
        token   = _access_token(entry)
        masked  = (token[:10] + "...") if len(token) > 10 else "—"
        expired = _is_expired(entry)
        status  = "EXPIRED" if expired else "OK     "
        print(f"  {status}  {name:<28} {env_var:<28} expires={_exp_str(entry)}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="pi-mono OAuth bridge")
    parser.add_argument("--check", action="store_true", help="Status only, no writes")
    parser.add_argument("--force", action="store_true", help="Refresh all tokens even if valid")
    parser.add_argument("--auth-file", type=Path, help="Override auth.json path")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help=f"Path to .env to upsert pi keys into (default: {_DEFAULT_ENV})",
    )
    args = parser.parse_args()

    env_out = args.env_file.resolve() if args.env_file else _DEFAULT_ENV

    auth_path = args.auth_file or find_auth_file()
    if not auth_path:
        print("auth.json not found. Run `pi login` first.")
        return

    print(f"auth.json: {auth_path}\n")

    if args.check:
        raw = json.loads(auth_path.read_text(encoding="utf-8"))
        print("Token status:")
        print_status(raw)
        return

    providers = load_and_refresh(auth_path, force=args.force)

    print("Token status:")
    print_status(providers)
    print()

    write_to_env(providers, env_out)


if __name__ == "__main__":
    main()
