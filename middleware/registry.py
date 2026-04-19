"""
Model registry — maps model names to provider + credentials + metadata.
Loaded once at startup from environment variables.

Style mirrors pi-mono (github.com/badlogic/pi-mono):
  - Full canonical model IDs as primary keys
  - Short aliases pointing to the same entries
  - Rich metadata: display name, reasoning, vision, context/output limits
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field


@dataclass
class ModelEntry:
    provider: str           # "anthropic" | "openai" | "gemini"
    model_id: str           # exact ID sent to the provider
    api_key: str
    name: str = ""          # human-readable display name
    reasoning: bool = False # supports extended thinking / chain-of-thought
    vision: bool = False    # accepts image inputs
    context_window: int = 200_000
    max_tokens: int = 8192
    base_url: str | None = None
    extra_headers: dict = field(default_factory=dict)


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def build_registry() -> dict[str, ModelEntry]:
    reg: dict[str, ModelEntry] = {}

    def add(aliases: list[str], entry: ModelEntry) -> None:
        for a in aliases:
            reg[a] = entry

    # ── Claude via GitHub Copilot ─────────────────────────────────────────────
    cop_key = _env("GITHUB_COPILOT_TOKEN")
    cop_hdrs = {"Editor-Version": "vscode/1.90.0", "Copilot-Integration-Id": "vscode-chat"}
    if cop_key:
        add(["claude-opus-4-5", "claude-opus", "opus"], ModelEntry(
            provider="openai", model_id="claude-opus-4-5", api_key=cop_key,
            base_url="https://api.githubcopilot.com",
            name="Claude Opus 4.5 (Copilot)", reasoning=True, vision=True,
            context_window=200_000, max_tokens=32_000,
            extra_headers=cop_hdrs,
        ))
        add(["claude-sonnet-4-5", "claude-sonnet", "claude", "sonnet"], ModelEntry(
            provider="openai", model_id="claude-sonnet-4-5", api_key=cop_key,
            base_url="https://api.githubcopilot.com",
            name="Claude Sonnet 4.5 (Copilot)", reasoning=True, vision=True,
            context_window=200_000, max_tokens=16_000,
            extra_headers=cop_hdrs,
        ))
        add(["claude-haiku-4-5-20251001", "claude-haiku", "haiku"], ModelEntry(
            provider="openai", model_id="claude-haiku-4-5-20251001", api_key=cop_key,
            base_url="https://api.githubcopilot.com",
            name="Claude Haiku 4.5 (Copilot)", vision=True,
            context_window=200_000, max_tokens=8_192,
            extra_headers=cop_hdrs,
        ))
        add(["claude-3-5-sonnet-20241022", "claude-3-5-sonnet"], ModelEntry(
            provider="openai", model_id="claude-3-5-sonnet-20241022", api_key=cop_key,
            base_url="https://api.githubcopilot.com",
            name="Claude 3.5 Sonnet (Copilot)", vision=True,
            context_window=200_000, max_tokens=8_192,
            extra_headers=cop_hdrs,
        ))
        add(["claude-3-5-haiku-20241022", "claude-3-5-haiku"], ModelEntry(
            provider="openai", model_id="claude-3-5-haiku-20241022", api_key=cop_key,
            base_url="https://api.githubcopilot.com",
            name="Claude 3.5 Haiku (Copilot)", vision=True,
            context_window=200_000, max_tokens=8_192,
            extra_headers=cop_hdrs,
        ))

    # ── OpenAI / Codex (via Copilot) ─────────────────────────────────────────

    # ── GitHub Copilot (GPT-4o) ───────────────────────────────────────────────
    if cop_key:
        add(["github-copilot", "copilot"], ModelEntry(
            provider="openai", model_id="gpt-4o", api_key=cop_key,
            base_url="https://api.githubcopilot.com",
            name="GitHub Copilot (GPT-4o)", vision=True,
            context_window=128_000, max_tokens=16_384,
            extra_headers=cop_hdrs,
        ))

    # ── Groq ──────────────────────────────────────────────────────────────────
    groq = _env("GROQ_API_KEY")
    if groq:
        add(["llama-3.3-70b-versatile", "llama-3.3-70b", "llama", "groq"], ModelEntry(
            provider="openai", model_id="llama-3.3-70b-versatile", api_key=groq,
            base_url="https://api.groq.com/openai/v1",
            name="Llama 3.3 70B (Groq)",
            context_window=128_000, max_tokens=32_768,
        ))
        add(["llama-3.1-8b-instant", "llama-fast"], ModelEntry(
            provider="openai", model_id="llama-3.1-8b-instant", api_key=groq,
            base_url="https://api.groq.com/openai/v1",
            name="Llama 3.1 8B Instant (Groq)",
            context_window=128_000, max_tokens=8_192,
        ))
        add(["deepseek-r1-groq", "deepseek-r1-distill-llama-70b"], ModelEntry(
            provider="openai", model_id="deepseek-r1-distill-llama-70b", api_key=groq,
            base_url="https://api.groq.com/openai/v1",
            name="DeepSeek R1 Distill (Groq)", reasoning=True,
            context_window=128_000, max_tokens=16_384,
        ))

    # ── Cerebras ──────────────────────────────────────────────────────────────
    cbr = _env("CEREBRAS_API_KEY")
    if cbr:
        add(["llama-3.3-70b-cerebras", "cerebras", "cerebras-fast"], ModelEntry(
            provider="openai", model_id="llama-3.3-70b", api_key=cbr,
            base_url="https://api.cerebras.ai/v1",
            name="Llama 3.3 70B (Cerebras)",
            context_window=128_000, max_tokens=8_192,
        ))
        add(["qwen-3-32b-cerebras", "qwen-cerebras"], ModelEntry(
            provider="openai", model_id="qwen-3-32b", api_key=cbr,
            base_url="https://api.cerebras.ai/v1",
            name="Qwen3 32B (Cerebras)", reasoning=True,
            context_window=32_000, max_tokens=16_000,
        ))

    # ── OpenRouter ────────────────────────────────────────────────────────────
    or_key = _env("OPENROUTER_API_KEY")
    or_hdrs = {"HTTP-Referer": "http://localhost", "X-Title": "ai-router"}
    if or_key:
        add(["auto", "openrouter"], ModelEntry(
            provider="openai", model_id="auto", api_key=or_key,
            base_url="https://openrouter.ai/api/v1",
            name="Auto (OpenRouter)", extra_headers=or_hdrs,
            context_window=200_000, max_tokens=16_384,
        ))
        add(["deepseek/deepseek-chat", "deepseek", "deepseek-v3"], ModelEntry(
            provider="openai", model_id="deepseek/deepseek-chat", api_key=or_key,
            base_url="https://openrouter.ai/api/v1",
            name="DeepSeek V3 (OpenRouter)", extra_headers=or_hdrs,
            context_window=163_840, max_tokens=8_192,
        ))
        add(["deepseek/deepseek-r1", "deepseek-r1"], ModelEntry(
            provider="openai", model_id="deepseek/deepseek-r1", api_key=or_key,
            base_url="https://openrouter.ai/api/v1",
            name="DeepSeek R1 (OpenRouter)", reasoning=True, extra_headers=or_hdrs,
            context_window=163_840, max_tokens=8_192,
        ))
        add(["qwen/qwen-2.5-72b-instruct", "qwen", "qwen-2.5"], ModelEntry(
            provider="openai", model_id="qwen/qwen-2.5-72b-instruct", api_key=or_key,
            base_url="https://openrouter.ai/api/v1",
            name="Qwen 2.5 72B (OpenRouter)", extra_headers=or_hdrs,
            context_window=131_072, max_tokens=8_192,
        ))
        add(["mistralai/mistral-7b-instruct:free", "free", "free-mistral"], ModelEntry(
            provider="openai", model_id="mistralai/mistral-7b-instruct:free", api_key=or_key,
            base_url="https://openrouter.ai/api/v1",
            name="Mistral 7B (free)", extra_headers=or_hdrs,
            context_window=32_768, max_tokens=4_096,
        ))
        add(["meta-llama/llama-3.1-8b-instruct:free", "free-llama"], ModelEntry(
            provider="openai", model_id="meta-llama/llama-3.1-8b-instruct:free", api_key=or_key,
            base_url="https://openrouter.ai/api/v1",
            name="Llama 3.1 8B (free)", extra_headers=or_hdrs,
            context_window=131_072, max_tokens=8_192,
        ))

    # ── ZAI ───────────────────────────────────────────────────────────────────
    zai = _env("ZAI_API_KEY")
    if zai:
        add(["glm-4-plus", "zai"], ModelEntry(
            provider="openai", model_id="glm-4-plus", api_key=zai,
            base_url="https://open.bigmodel.cn/api/paas/v4",
            name="GLM-4 Plus (ZAI)", vision=True,
            context_window=128_000, max_tokens=4_096,
        ))

    # ── Kilo ──────────────────────────────────────────────────────────────────
    kilo = _env("KILO_API_KEY")
    if kilo:
        add(["kilo"], ModelEntry(
            provider="openai", model_id="kilo-default", api_key=kilo,
            base_url="https://api.kilo.ai/v1",
            name="Kilo Default",
            context_window=200_000, max_tokens=8_192,
        ))

    # ── OpenCode ──────────────────────────────────────────────────────────────
    oc_key = _env("OPENCODE_API_KEY")
    if oc_key:
        add(["opencode"], ModelEntry(
            provider="openai", model_id="opencode-default", api_key=oc_key,
            base_url=_env("OPENCODE_BASE_URL") or "https://api.opencode.ai/v1",
            name="OpenCode Default",
            context_window=200_000, max_tokens=8_192,
        ))

    # ── OpenCode Zen / MiniMax ────────────────────────────────────────────────
    zen_key = _env("OPENCODE_ZEN_API_KEY")
    if zen_key:
        add(["MiniMax-Text-01", "opencode-zen", "minimax"], ModelEntry(
            provider="openai", model_id="MiniMax-Text-01", api_key=zen_key,
            base_url=_env("OPENCODE_ZEN_BASE_URL") or "https://api.minimaxi.chat/v1",
            name="MiniMax Text-01 (OpenCode Zen)",
            context_window=1_000_000, max_tokens=8_192,
        ))

    return reg


# Singleton loaded at startup via init()
REGISTRY: dict[str, ModelEntry] = {}


def init(env_path: str | None = None) -> None:
    global REGISTRY
    if env_path:
        _load_dotenv(env_path)
    REGISTRY = build_registry()


def get(model_name: str) -> ModelEntry | None:
    return REGISTRY.get(model_name) or REGISTRY.get(model_name.lower())


def list_models() -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for alias, e in sorted(REGISTRY.items()):
        canonical = (e.provider, e.model_id, e.base_url or "")
        is_primary = canonical not in seen
        seen.add(canonical)
        out.append({
            "id": alias,
            "object": "model",
            "owned_by": e.provider,
            "name": e.name,
            "reasoning": e.reasoning,
            "vision": e.vision,
            "context_window": e.context_window,
            "max_tokens": e.max_tokens,
            "primary": is_primary,
        })
    return out


def _load_dotenv(path: str) -> None:
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip()
            if k and v and "REPLACE_ME" not in v:
                os.environ.setdefault(k, v)
