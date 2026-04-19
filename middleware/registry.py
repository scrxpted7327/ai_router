"""
Model registry — canonical catalog + alias index + metadata enrichment.

The registry follows the same broad shape as OpenCode's model handling:
  - canonical models are the source of truth
  - aliases resolve to canonical entries without creating duplicate visible rows
  - provider/model metadata can be enriched from models.dev and cached locally
  - provider/model filtering is applied after normalization
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import httpx

MODELS_DEV_URL = "https://models.dev/api.json"
MODELS_DEV_CACHE_TTL = 60 * 60
MODELS_DEV_CACHE_PATH = Path.home() / ".cache" / "ai_router" / "models.dev.json"
BEDROCK_PROVIDER_ID = "amazon-bedrock"


@dataclass(frozen=True)
class CatalogModel:
    provider_id: str
    provider_api: str
    provider_label: str
    model_id: str
    name: str
    aliases: tuple[str, ...] = ()
    reasoning: bool = False
    vision: bool = False
    context_window: int = 200_000
    max_tokens: int = 8192
    status: str = "stable"


@dataclass(frozen=True)
class ProviderConfig:
    id: str
    api: str
    label: str
    env_keys: tuple[str, ...]
    base_url: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    options: dict[str, str] = field(default_factory=dict)


@dataclass
class ModelEntry:
    provider: str
    model_id: str
    api_key: str
    name: str = ""
    reasoning: bool = False
    vision: bool = False
    context_window: int = 200_000
    max_tokens: int = 8192
    base_url: str | None = None
    extra_headers: dict = field(default_factory=dict)
    provider_label: str = ""
    provider_id: str = ""
    aliases: tuple[str, ...] = ()
    options: dict[str, str] = field(default_factory=dict)


@dataclass
class RegistryState:
    by_canonical_id: dict[str, ModelEntry] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _csv_set(raw: str) -> set[str]:
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def _normalise_key(value: str) -> str:
    return value.strip().lower()


def _provider_allowed(provider_id: str) -> bool:
    enabled = _csv_set(_env("MODEL_ENABLED_PROVIDERS"))
    disabled = _csv_set(_env("MODEL_DISABLED_PROVIDERS"))
    key = provider_id.lower()
    if enabled and key not in enabled:
        return False
    if key in disabled:
        return False
    return True


def _model_allowed(model_id: str) -> bool:
    whitelist = _csv_set(_env("MODEL_WHITELIST"))
    blacklist = _csv_set(_env("MODEL_BLACKLIST"))
    key = model_id.lower()
    if whitelist and key not in whitelist:
        return False
    if key in blacklist:
        return False
    return True


def _provider_model_allowed(provider_id: str, model_id: str) -> bool:
    provider_key = provider_id.upper().replace("-", "_")
    whitelist = _csv_set(_env(f"{provider_key}_MODEL_WHITELIST"))
    blacklist = _csv_set(_env(f"{provider_key}_MODEL_BLACKLIST"))
    key = model_id.lower()
    if whitelist and key not in whitelist:
        return False
    if key in blacklist:
        return False
    return True


def _load_models_dev() -> dict:
    try:
        MODELS_DEV_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return {}

    cached = _read_models_dev_cache()
    now = int(time.time())
    if cached and now - int(cached.get("fetched_at", 0)) < MODELS_DEV_CACHE_TTL:
        return cached.get("payload", {}) or {}

    try:
        with httpx.Client(timeout=1.5, follow_redirects=True) as client:
            response = client.get(MODELS_DEV_URL)
            response.raise_for_status()
            payload = response.json()
        MODELS_DEV_CACHE_PATH.write_text(
            json.dumps({"fetched_at": now, "payload": payload}),
            encoding="utf-8",
        )
        return payload
    except Exception:
        return cached.get("payload", {}) if cached else {}


def _read_models_dev_cache() -> dict:
    try:
        if not MODELS_DEV_CACHE_PATH.exists():
            return {}
        return json.loads(MODELS_DEV_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _index_models_dev(payload: dict) -> dict[str, dict]:
    if not isinstance(payload, dict):
        return {}
    providers = payload.get("providers")
    if not isinstance(providers, dict):
        return {}

    index: dict[str, dict] = {}
    for provider_data in providers.values():
        if not isinstance(provider_data, dict):
            continue
        models = provider_data.get("models")
        if not isinstance(models, dict):
            continue
        for model_id, model_data in models.items():
            if isinstance(model_data, dict) and model_id not in index:
                index[model_id] = model_data
    return index


def _bool_capability(model_data: dict, *keys: str) -> bool:
    capabilities = model_data.get("capabilities")
    if isinstance(capabilities, dict):
        for key in keys:
            if bool(capabilities.get(key)):
                return True
    for key in keys:
        if bool(model_data.get(key)):
            return True
    return False


def _int_value(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _enrich_model(model: CatalogModel, models_dev: dict[str, dict]) -> CatalogModel:
    model_data = models_dev.get(model.model_id)
    if not model_data and "/" in model.model_id:
        model_data = models_dev.get(model.model_id.split("/", 1)[-1])
    if not model_data:
        return model

    context_window = model.context_window
    limit = model_data.get("limit")
    if isinstance(limit, dict):
        context_window = _int_value(limit.get("context") or limit.get("input"), context_window)
        max_tokens = _int_value(limit.get("output"), model.max_tokens)
    else:
        max_tokens = model.max_tokens

    name = str(model_data.get("name") or model.name).strip() or model.name
    status = str(model_data.get("status") or model.status).strip() or model.status

    return replace(
        model,
        name=name,
        reasoning=model.reasoning or _bool_capability(model_data, "reasoning"),
        vision=model.vision or _bool_capability(model_data, "vision", "image_input"),
        context_window=context_window,
        max_tokens=max_tokens,
        status=status,
    )


def _bedrock_region_prefix(region: str) -> str:
    region = (region or "").lower()
    if region.startswith("eu-"):
        return "eu"
    if region.startswith("ap-"):
        return "apac"
    if region.startswith("jp-"):
        return "jp"
    if region.startswith("au-"):
        return "au"
    return "us"


def _maybe_prefix_bedrock_model_id(model_id: str, region: str) -> str:
    lowered = model_id.lower()
    if lowered.startswith(("global.", "us.", "eu.", "jp.", "apac.", "au.", "arn:")):
        return model_id
    if lowered.startswith("anthropic."):
        return f"{_bedrock_region_prefix(region)}.{model_id}"
    return model_id


def _openai_provider() -> tuple[ProviderConfig, tuple[CatalogModel, ...]] | None:
    if not _env("OPENAI_API_KEY"):
        return None
    provider = ProviderConfig("openai", "openai", "openai", ("OPENAI_API_KEY",), base_url="https://api.openai.com/v1")
    models = (
        CatalogModel(provider.id, provider.api, provider.label, "gpt-4.1",          "GPT-4.1",       ("gpt-4.1",),       False, True,  1_047_576, 32_768),
        CatalogModel(provider.id, provider.api, provider.label, "gpt-4.1-mini",     "GPT-4.1 Mini",  ("gpt-4.1-mini",),  False, True,  1_047_576, 32_768),
        CatalogModel(provider.id, provider.api, provider.label, "gpt-4.1-nano",     "GPT-4.1 Nano",  ("gpt-4.1-nano",),  False, True,  1_047_576, 32_768),
        CatalogModel(provider.id, provider.api, provider.label, "gpt-4o",           "GPT-4o",        ("gpt4o",),         False, True,  128_000, 16_384),
        CatalogModel(provider.id, provider.api, provider.label, "gpt-4o-mini",      "GPT-4o Mini",   ("gpt4o-mini",),    False, True,  128_000, 16_384),
        CatalogModel(provider.id, provider.api, provider.label, "o1",               "o1",            ("openai-o1",),     True,  True,  200_000, 100_000),
        CatalogModel(provider.id, provider.api, provider.label, "o1-mini",          "o1 Mini",       ("openai-o1-mini",),True,  False, 128_000, 65_536),
        CatalogModel(provider.id, provider.api, provider.label, "o3",               "o3",            ("openai-o3",),     True,  True,  200_000, 100_000),
        CatalogModel(provider.id, provider.api, provider.label, "o3-mini",          "o3 Mini",       ("openai-o3-mini",),True,  False, 200_000, 100_000),
        CatalogModel(provider.id, provider.api, provider.label, "o4-mini",          "o4 Mini",       ("openai-o4-mini",),True,  True,  200_000, 100_000),
        CatalogModel(provider.id, provider.api, provider.label, "gpt-5",            "GPT-5",         ("openai-gpt5",),   True,  True,  1_000_000, 32_768),
    )
    return provider, models


def _mistral_provider() -> tuple[ProviderConfig, tuple[CatalogModel, ...]] | None:
    if not _env("MISTRAL_API_KEY"):
        return None
    provider = ProviderConfig("mistral", "openai", "mistral", ("MISTRAL_API_KEY",), base_url="https://api.mistral.ai/v1")
    models = (
        CatalogModel(provider.id, provider.api, provider.label, "mistral-large-latest",  "Mistral Large",    ("mistral-large",),   False, False, 131_072, 4_096),
        CatalogModel(provider.id, provider.api, provider.label, "mistral-medium-latest", "Mistral Medium",   ("mistral-medium",),  False, False, 131_072, 4_096),
        CatalogModel(provider.id, provider.api, provider.label, "mistral-small-latest",  "Mistral Small",    ("mistral-small",),   False, False, 131_072, 4_096),
        CatalogModel(provider.id, provider.api, provider.label, "codestral-latest",      "Codestral",        ("codestral",),       False, False, 256_000, 4_096),
        CatalogModel(provider.id, provider.api, provider.label, "mistral-nemo",          "Mistral Nemo",     ("mistral-nemo",),    False, False, 131_072, 4_096),
        CatalogModel(provider.id, provider.api, provider.label, "pixtral-large-latest",  "Pixtral Large",    ("pixtral-large",),   False, True,  131_072, 4_096),
    )
    return provider, models


def _deepseek_provider() -> tuple[ProviderConfig, tuple[CatalogModel, ...]] | None:
    if not _env("DEEPSEEK_API_KEY"):
        return None
    provider = ProviderConfig("deepseek", "openai", "deepseek", ("DEEPSEEK_API_KEY",), base_url="https://api.deepseek.com/v1")
    models = (
        CatalogModel(provider.id, provider.api, provider.label, "deepseek-chat",     "DeepSeek V3",   ("deepseek-v3-direct", "deepseek-direct"), False, False, 163_840, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "deepseek-reasoner", "DeepSeek R1",   ("deepseek-r1-direct",),                   True,  False, 163_840, 8_192),
    )
    return provider, models


def _xai_provider() -> tuple[ProviderConfig, tuple[CatalogModel, ...]] | None:
    if not _env("XAI_API_KEY"):
        return None
    provider = ProviderConfig("xai", "openai", "xai", ("XAI_API_KEY",), base_url="https://api.x.ai/v1")
    models = (
        CatalogModel(provider.id, provider.api, provider.label, "grok-3",        "Grok 3",        ("grok",),        True,  False, 131_072, 131_072),
        CatalogModel(provider.id, provider.api, provider.label, "grok-3-mini",   "Grok 3 Mini",   ("grok-mini",),   True,  False, 131_072, 131_072),
        CatalogModel(provider.id, provider.api, provider.label, "grok-3-fast",   "Grok 3 Fast",   ("grok-fast",),   False, False, 131_072, 131_072),
    )
    return provider, models


def _together_provider() -> tuple[ProviderConfig, tuple[CatalogModel, ...]] | None:
    if not _env("TOGETHER_API_KEY"):
        return None
    provider = ProviderConfig("together", "openai", "together", ("TOGETHER_API_KEY",), base_url="https://api.together.xyz/v1")
    models = (
        CatalogModel(provider.id, provider.api, provider.label, "meta-llama/Llama-3.3-70B-Instruct-Turbo",   "Llama 3.3 70B (Together)",    ("llama-together",),         False, False, 131_072, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", "Llama 3.1 405B (Together)", ("llama-405b",),           False, False, 130_815, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "Llama 4 Maverick (Together)", ("llama-4-maverick",), False, True,  524_288, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "meta-llama/Llama-4-Scout-17B-16E-Instruct",  "Llama 4 Scout (Together)",    ("llama-4-scout",),          False, True,  131_072, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "deepseek-ai/DeepSeek-R1",                    "DeepSeek R1 (Together)",      ("deepseek-r1-together",),   True,  False, 163_840, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "Qwen/QwQ-32B-Preview",                       "QwQ 32B (Together)",          ("qwq-together",),           True,  False, 32_768,  8_192),
        CatalogModel(provider.id, provider.api, provider.label, "mistralai/Mistral-7B-Instruct-v0.3",         "Mistral 7B (Together)",       ("mistral-7b-together",),    False, False, 32_768,  4_096),
    )
    return provider, models


def _perplexity_provider() -> tuple[ProviderConfig, tuple[CatalogModel, ...]] | None:
    if not _env("PERPLEXITY_API_KEY"):
        return None
    provider = ProviderConfig("perplexity", "openai", "perplexity", ("PERPLEXITY_API_KEY",), base_url="https://api.perplexity.ai")
    models = (
        CatalogModel(provider.id, provider.api, provider.label, "sonar-pro",              "Sonar Pro",              ("sonar-pro",),          False, False, 200_000, 8_000),
        CatalogModel(provider.id, provider.api, provider.label, "sonar",                  "Sonar",                  ("sonar",),              False, False, 200_000, 8_000),
        CatalogModel(provider.id, provider.api, provider.label, "sonar-reasoning-pro",    "Sonar Reasoning Pro",    ("sonar-reason-pro",),   True,  False, 128_000, 8_000),
        CatalogModel(provider.id, provider.api, provider.label, "sonar-reasoning",        "Sonar Reasoning",        ("sonar-reason",),       True,  False, 128_000, 8_000),
        CatalogModel(provider.id, provider.api, provider.label, "sonar-deep-research",    "Sonar Deep Research",    ("sonar-research",),     True,  False, 128_000, 8_000),
    )
    return provider, models


def _fireworks_provider() -> tuple[ProviderConfig, tuple[CatalogModel, ...]] | None:
    if not _env("FIREWORKS_API_KEY"):
        return None
    provider = ProviderConfig("fireworks", "openai", "fireworks", ("FIREWORKS_API_KEY",), base_url="https://api.fireworks.ai/inference/v1")
    models = (
        CatalogModel(provider.id, provider.api, provider.label, "accounts/fireworks/models/llama-v3p3-70b-instruct",    "Llama 3.3 70B (Fireworks)",    ("llama-fireworks",),         False, False, 131_072, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "accounts/fireworks/models/llama-v3p1-405b-instruct",   "Llama 3.1 405B (Fireworks)",   ("llama-405b-fireworks",),    False, False, 131_072, 4_096),
        CatalogModel(provider.id, provider.api, provider.label, "accounts/fireworks/models/deepseek-r1",                "DeepSeek R1 (Fireworks)",      ("deepseek-r1-fireworks",),   True,  False, 163_840, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "accounts/fireworks/models/qwen3-235b-a22b",            "Qwen 3 235B (Fireworks)",      ("qwen3-235b-fireworks",),    True,  False, 131_072, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "accounts/fireworks/models/mixtral-8x22b-instruct",     "Mixtral 8x22B (Fireworks)",    ("mixtral-fireworks",),       False, False, 65_536,  4_096),
    )
    return provider, models


def _cohere_provider() -> tuple[ProviderConfig, tuple[CatalogModel, ...]] | None:
    if not _env("COHERE_API_KEY"):
        return None
    provider = ProviderConfig("cohere", "openai", "cohere", ("COHERE_API_KEY",), base_url="https://api.cohere.ai/compatibility/v1")
    models = (
        CatalogModel(provider.id, provider.api, provider.label, "command-a-03-2025",       "Command A",           ("command-a", "cohere"),         False, False, 256_000, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "command-r-plus-08-2024",  "Command R+",          ("command-r-plus",),             False, False, 128_000, 4_096),
        CatalogModel(provider.id, provider.api, provider.label, "command-r-08-2024",       "Command R",           ("command-r",),                  False, False, 128_000, 4_096),
        CatalogModel(provider.id, provider.api, provider.label, "command-r7b-12-2024",     "Command R 7B",        ("command-r7b",),                False, False, 128_000, 4_096),
    )
    return provider, models


def _copilot_provider() -> tuple[ProviderConfig, tuple[CatalogModel, ...]] | None:
    token = _env("GITHUB_COPILOT_TOKEN")
    if not token:
        return None
    provider = ProviderConfig(
        id="github-copilot",
        api="openai",
        label="github-copilot",
        env_keys=("GITHUB_COPILOT_TOKEN",),
        base_url="https://api.githubcopilot.com",
        extra_headers={
            "Editor-Version": "vscode/1.90.0",
            "Copilot-Integration-Id": "vscode-chat",
        },
    )
    models = (
        CatalogModel(provider.id, provider.api, provider.label, "gpt-5.4-mini",            "GPT-5.4 Mini (Copilot)",      ("copilot-5.4",),               True,  True,  1_000_000, 32_768),
        CatalogModel(provider.id, provider.api, provider.label, "gpt-5.3-codex",           "GPT-5.3 Codex (Copilot)",     ("copilot-codex",),             True,  True,  1_000_000, 100_000),
        CatalogModel(provider.id, provider.api, provider.label, "gpt-5.2-codex",           "GPT-5.2 Codex (Copilot)",     ("copilot-5.2-codex",),         True,  True,  1_000_000, 100_000),
        CatalogModel(provider.id, provider.api, provider.label, "gpt-5.2",                 "GPT-5.2 (Copilot)",           ("copilot-5.2",),               False, True,  1_000_000, 32_768),
        CatalogModel(provider.id, provider.api, provider.label, "gpt-5-mini",              "GPT-5 Mini (Copilot)",        ("copilot-5-mini",),            False, True,  1_000_000, 16_384),
        CatalogModel(provider.id, provider.api, provider.label, "claude-opus-4-7",         "Claude Opus 4.7 (Copilot)",   ("claude-opus", "opus"),        True,  True,  200_000, 32_000),
        CatalogModel(provider.id, provider.api, provider.label, "claude-opus-4-5",         "Claude Opus 4.5 (Copilot)",   ("claude-opus-4-5-copilot",),   True,  True,  200_000, 32_000),
        CatalogModel(provider.id, provider.api, provider.label, "claude-sonnet-4-6",       "Claude Sonnet 4.6 (Copilot)", ("claude-sonnet", "claude", "sonnet"), False, True, 200_000, 16_000),
        CatalogModel(provider.id, provider.api, provider.label, "claude-sonnet-4-5",       "Claude Sonnet 4.5 (Copilot)", ("claude-sonnet-4-5-copilot",), True,  True,  200_000, 16_000),
        CatalogModel(provider.id, provider.api, provider.label, "claude-sonnet-4",         "Claude Sonnet 4 (Copilot)",   ("claude-sonnet-4-copilot",),   False, True,  200_000, 16_000),
        CatalogModel(provider.id, provider.api, provider.label, "claude-haiku-4-5-20251001","Claude Haiku 4.5 (Copilot)", ("claude-haiku", "haiku"),      False, True,  200_000,  8_192),
        CatalogModel(provider.id, provider.api, provider.label, "gemini-3.1-pro-preview",  "Gemini 3.1 Pro (Copilot)",    ("gemini-3.1-copilot",),        True,  True,  1_048_576, 65_536),
        CatalogModel(provider.id, provider.api, provider.label, "gemini-3-flash",          "Gemini 3 Flash (Copilot)",    ("gemini-3-copilot",),          False, True,  1_048_576, 32_768),
        CatalogModel(provider.id, provider.api, provider.label, "gemini-2.5-pro",          "Gemini 2.5 Pro (Copilot)",    ("gemini-pro-copilot",),        True,  True,  1_048_576, 65_536),
        CatalogModel(provider.id, provider.api, provider.label, "gpt-4.1",                 "GPT-4.1 (Copilot)",           ("gpt-4.1-copilot",),           False, True,  1_047_576, 32_768),
        CatalogModel(provider.id, provider.api, provider.label, "gpt-4o",                  "GPT-4o (Copilot)",            ("github-copilot", "copilot"),  False, True,  128_000, 16_384),
        CatalogModel(provider.id, provider.api, provider.label, "grok-code-fast-1",        "Grok Code Fast (Copilot)",    ("grok-copilot",),              False, False, 131_072,  8_192),
    )
    return provider, models


def _groq_provider() -> tuple[ProviderConfig, tuple[CatalogModel, ...]] | None:
    if not _env("GROQ_API_KEY"):
        return None
    provider = ProviderConfig("groq", "openai", "groq", ("GROQ_API_KEY",), base_url="https://api.groq.com/openai/v1")
    models = (
        CatalogModel(provider.id, provider.api, provider.label, "meta-llama/llama-4-maverick-17b-128e-instruct", "Llama 4 Maverick (Groq)", ("llama-4-maverick-groq",), False, True,  1_047_576, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "meta-llama/llama-4-scout-17b-16e-instruct",     "Llama 4 Scout (Groq)",    ("llama-4-scout-groq",),    False, True,  131_072,   8_192),
        CatalogModel(provider.id, provider.api, provider.label, "llama-3.3-70b-versatile",       "Llama 3.3 70B (Groq)",          ("llama-3.3-70b", "llama", "groq"), False, False, 128_000, 32_768),
        CatalogModel(provider.id, provider.api, provider.label, "llama-3.2-90b-vision-preview",  "Llama 3.2 90B Vision (Groq)",   ("llama-3.2-90b-groq",),            False, True,  128_000,  8_192),
        CatalogModel(provider.id, provider.api, provider.label, "llama-3.2-11b-vision-preview",  "Llama 3.2 11B Vision (Groq)",   ("llama-3.2-11b-groq",),            False, True,  128_000,  8_192),
        CatalogModel(provider.id, provider.api, provider.label, "llama-3.2-3b-preview",          "Llama 3.2 3B (Groq)",           ("llama-3.2-3b-groq",),             False, False,  16_384,  8_192),
        CatalogModel(provider.id, provider.api, provider.label, "llama-3.1-8b-instant",           "Llama 3.1 8B Instant (Groq)",   ("llama-fast",),                    False, False, 128_000,  8_192),
        CatalogModel(provider.id, provider.api, provider.label, "llama3-70b-8192",                "Llama 3 70B (Groq)",            ("llama3-70b-groq",),               False, False,   8_192,  8_192),
        CatalogModel(provider.id, provider.api, provider.label, "deepseek-r1-distill-llama-70b",  "DeepSeek R1 Distill (Groq)",    ("deepseek-r1-groq",),              True,  False, 128_000, 16_384),
        CatalogModel(provider.id, provider.api, provider.label, "qwen-qwq-32b",                   "QwQ 32B (Groq)",                ("qwq-groq",),                      True,  False, 131_072,  8_192),
        CatalogModel(provider.id, provider.api, provider.label, "qwen/qwen3-32b",                 "Qwen3 32B (Groq)",              ("qwen3-groq",),                    True,  False,  32_000, 16_000),
        CatalogModel(provider.id, provider.api, provider.label, "moonshotai/kimi-k2-instruct-0905", "Kimi K2 Instruct (Groq)",    ("kimi-groq",),                     False, False, 131_072,  8_192),
        CatalogModel(provider.id, provider.api, provider.label, "openai/gpt-oss-120b",            "GPT-OSS 120B (Groq)",           ("gpt-oss-groq",),                  False, False, 131_072, 16_384),
        CatalogModel(provider.id, provider.api, provider.label, "groq/compound",                  "Compound (Groq)",               ("compound-groq",),                 False, True,  128_000,  8_192),
        CatalogModel(provider.id, provider.api, provider.label, "mistral-saba-24b",               "Mistral Saba 24B (Groq)",       ("mistral-saba-groq",),             False, False,  32_768,  4_096),
        CatalogModel(provider.id, provider.api, provider.label, "gemma2-9b-it",                   "Gemma 2 9B (Groq)",             ("gemma2-groq",),                   False, False,   8_192,  8_192),
    )
    return provider, models


def _cerebras_provider() -> tuple[ProviderConfig, tuple[CatalogModel, ...]] | None:
    if not _env("CEREBRAS_API_KEY"):
        return None
    provider = ProviderConfig("cerebras", "openai", "cerebras", ("CEREBRAS_API_KEY",), base_url="https://api.cerebras.ai/v1", extra_headers={"X-Cerebras-3rd-Party-Integration": "ai-router"})
    models = (
        CatalogModel(provider.id, provider.api, provider.label, "llama-3.3-70b",       "Llama 3.3 70B (Cerebras)",        ("llama-3.3-70b-cerebras", "cerebras", "cerebras-fast"), False, False, 128_000, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "llama3.1-70b",        "Llama 3.1 70B (Cerebras)",        ("llama3.1-70b-cerebras",),                              False, False, 128_000, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "llama3.1-8b",         "Llama 3.1 8B (Cerebras)",         ("llama3.1-8b-cerebras",),                               False, False, 128_000, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "llama-4-scout-17b-16e-instruct", "Llama 4 Scout (Cerebras)", ("llama-4-scout-cerebras",),                         False, True,  131_072, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "qwen/qwen3-32b",      "Qwen3 32B (Cerebras)",            ("qwen-3-32b-cerebras", "qwen-cerebras"),                True,  False,  32_000, 16_000),
        CatalogModel(provider.id, provider.api, provider.label, "qwen/qwen3-235b-a22b-instruct-2507", "Qwen3 235B (Cerebras)", ("qwen-235b-cerebras",),                         True,  False, 131_072,  8_192),
        CatalogModel(provider.id, provider.api, provider.label, "deepseek-r1-distill-llama-70b", "DeepSeek R1 Distill (Cerebras)", ("deepseek-r1-cerebras",),                     True,  False, 128_000, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "gpt-oss-120b",        "GPT-OSS 120B (Cerebras)",         ("gpt-oss-cerebras",),                                   False, False, 131_072, 16_384),
        CatalogModel(provider.id, provider.api, provider.label, "zai-glm-4.7",         "GLM-4.7 (Cerebras)",              ("glm-cerebras",),                                       False, True,  128_000,  4_096),
    )
    return provider, models


def _auto_routing_provider() -> tuple[ProviderConfig, tuple[CatalogModel, ...]] | None:
    provider = ProviderConfig(
        "auto-routing",
        "openai",
        "scrxpted",
        (),
        base_url="INTERNAL",
    )
    models = (
        CatalogModel(provider.id, provider.api, provider.label, "scrxpted/auto-free",    "Auto Free Router",     ("auto-free",),    False, True,  200_000, 16_384),
        CatalogModel(provider.id, provider.api, provider.label, "scrxpted/auto-premium", "Auto Premium Router",  ("auto-premium",), True,  True,  1_000_000, 32_768),
        CatalogModel(provider.id, provider.api, provider.label, "scrxpted/auto-max",     "Auto Max Router",      ("auto-max",),     True,  True,  1_000_000, 100_000),
    )
    return provider, models


def _openrouter_provider() -> tuple[ProviderConfig, tuple[CatalogModel, ...]] | None:
    if not _env("OPENROUTER_API_KEY"):
        return None
    provider = ProviderConfig(
        "openrouter",
        "openai",
        "openrouter",
        ("OPENROUTER_API_KEY",),
        base_url="https://openrouter.ai/api/v1",
        extra_headers={"HTTP-Referer": "http://localhost", "X-Title": "ai-router"},
    )
    models = (
        CatalogModel(provider.id, provider.api, provider.label, "auto",                                    "Auto (OpenRouter)",           ("openrouter",),           False, False, 200_000, 16_384),
        CatalogModel(provider.id, provider.api, provider.label, "anthropic/claude-opus-4-7",               "Claude Opus 4.7 (OR)",        ("claude-opus-or",),       True,  True,  200_000, 32_000),
        CatalogModel(provider.id, provider.api, provider.label, "anthropic/claude-sonnet-4-6",             "Claude Sonnet 4.6 (OR)",      ("claude-sonnet-or",),     False, True,  200_000, 16_000),
        CatalogModel(provider.id, provider.api, provider.label, "openai/gpt-4o",                           "GPT-4o (OR)",                 ("gpt-4o-or",),            False, True,  128_000, 16_384),
        CatalogModel(provider.id, provider.api, provider.label, "openai/o3",                               "o3 (OR)",                     ("o3-or",),                True,  True,  200_000, 100_000),
        CatalogModel(provider.id, provider.api, provider.label, "google/gemini-2.5-pro",                   "Gemini 2.5 Pro (OR)",         ("gemini-pro-or",),        True,  True,  1_048_576, 65_536),
        CatalogModel(provider.id, provider.api, provider.label, "google/gemini-2.5-flash",                 "Gemini 2.5 Flash (OR)",       ("gemini-flash-or",),      False, True,  1_048_576, 65_536),
        CatalogModel(provider.id, provider.api, provider.label, "deepseek/deepseek-chat",                  "DeepSeek V3 (OR)",            ("deepseek", "deepseek-v3"), False, False, 163_840, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "deepseek/deepseek-r1",                    "DeepSeek R1 (OR)",            ("deepseek-r1",),          True,  False, 163_840, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "x-ai/grok-3",                             "Grok 3 (OR)",                 ("grok-or",),              True,  False, 131_072, 131_072),
        CatalogModel(provider.id, provider.api, provider.label, "qwen/qwen3-235b-a22b",                    "Qwen 3 235B (OR)",            ("qwen3-or",),             True,  False, 131_072, 40_000),
        CatalogModel(provider.id, provider.api, provider.label, "qwen/qwen-2.5-72b-instruct",              "Qwen 2.5 72B (OR)",           ("qwen", "qwen-2.5"),      False, False, 131_072,  8_192),
        CatalogModel(provider.id, provider.api, provider.label, "meta-llama/llama-4-maverick",             "Llama 4 Maverick (OR)",       ("llama-4-maverick-or",),  False, True,  524_288,  8_192),
        CatalogModel(provider.id, provider.api, provider.label, "meta-llama/llama-4-scout",                "Llama 4 Scout (OR)",          ("llama-4-scout-or",),     False, True,  131_072,  8_192),
        CatalogModel(provider.id, provider.api, provider.label, "mistralai/mistral-large-2411",            "Mistral Large (OR)",          ("mistral-large-or",),     False, False, 131_072,  4_096),
        CatalogModel(provider.id, provider.api, provider.label, "meta-llama/llama-3.3-70b-instruct",       "Llama 3.3 70B (OR)",          ("llama-3.3-or",),         False, False, 128_000,  8_192),
        CatalogModel(provider.id, provider.api, provider.label, "meta-llama/llama-3.2-90b-vision-instruct","Llama 3.2 90B Vision (OR)",   ("llama-3.2-90b-or",),     False, True,  128_000,  8_192),
        CatalogModel(provider.id, provider.api, provider.label, "microsoft/phi-4",                         "Phi-4 (OR)",                  ("phi-4",),                False, False, 128_000, 16_384),
        CatalogModel(provider.id, provider.api, provider.label, "microsoft/phi-4-reasoning",               "Phi-4 Reasoning (OR)",        ("phi-4-reasoning",),      True,  False, 128_000, 32_768),
        CatalogModel(provider.id, provider.api, provider.label, "nvidia/llama-3.1-nemotron-ultra-253b-v1", "Nemotron Ultra (OR)",         ("nemotron",),             True,  False, 128_000, 32_768),
        CatalogModel(provider.id, provider.api, provider.label, "amazon/nova-pro-v1",                      "Amazon Nova Pro (OR)",        ("nova-pro",),             False, True,  300_000, 5_120),
        CatalogModel(provider.id, provider.api, provider.label, "amazon/nova-lite-v1",                     "Amazon Nova Lite (OR)",       ("nova-lite",),            False, True,  300_000, 5_120),
        CatalogModel(provider.id, provider.api, provider.label, "mistralai/mistral-7b-instruct:free",      "Mistral 7B (free)",           ("free", "free-mistral"),  False, False,  32_768,  4_096),
        CatalogModel(provider.id, provider.api, provider.label, "meta-llama/llama-3.1-8b-instruct:free",   "Llama 3.1 8B (free)",         ("free-llama",),           False, False, 131_072,  8_192),
        CatalogModel(provider.id, provider.api, provider.label, "google/gemma-3-27b-it:free",              "Gemma 3 27B (free)",          ("free-gemma",),           False, True,  131_072,  8_192),
    )
    return provider, models


def _zai_provider() -> tuple[ProviderConfig, tuple[CatalogModel, ...]] | None:
    if not _env("ZAI_API_KEY"):
        return None
    provider = ProviderConfig("zai", "openai", "zai", ("ZAI_API_KEY",), base_url="https://open.bigmodel.cn/api/paas/v4")
    models = (
        CatalogModel(provider.id, provider.api, provider.label, "glm-4-plus",         "GLM-4 Plus",           ("zai", "glm"),              False, True,  128_000,  4_096),
        CatalogModel(provider.id, provider.api, provider.label, "glm-4-air",          "GLM-4 Air",            ("glm-air",),                False, False, 128_000,  4_096),
        CatalogModel(provider.id, provider.api, provider.label, "glm-4-airx",         "GLM-4 AirX",           ("glm-airx",),               False, False, 128_000,  4_096),
        CatalogModel(provider.id, provider.api, provider.label, "glm-4-flash",        "GLM-4 Flash",          ("glm-flash", "zai-free"),   False, False, 128_000,  4_096),
        CatalogModel(provider.id, provider.api, provider.label, "glm-4-long",         "GLM-4 Long",           ("glm-long",),               False, False, 1_000_000, 4_096),
        CatalogModel(provider.id, provider.api, provider.label, "glm-4v-plus",        "GLM-4V Plus",          ("glm-4v",),                 False, True,  8_192,   4_096),
        CatalogModel(provider.id, provider.api, provider.label, "glm-z1-air",         "GLM-Z1 Air",           ("glm-z1",),                 True,  False, 128_000,  4_096),
        CatalogModel(provider.id, provider.api, provider.label, "glm-z1-flash",       "GLM-Z1 Flash",         ("glm-z1-flash",),           True,  False, 128_000,  4_096),
        CatalogModel(provider.id, provider.api, provider.label, "codegeex-4",         "CodeGeeX-4",           ("codegeex",),               False, False, 128_000,  4_096),
    )
    return provider, models


def _kilo_provider() -> tuple[ProviderConfig, tuple[CatalogModel, ...]] | None:
    if not _env("KILO_API_KEY"):
        return None
    provider = ProviderConfig("kilo", "openai", "kilo", ("KILO_API_KEY",), base_url="https://api.kilo.ai/api/gateway", extra_headers={"HTTP-Referer": "https://ai.scrxpted.cc/", "X-Title": "ai-router"})
    models = (
        CatalogModel(provider.id, provider.api, provider.label, "kilo-auto-free", "Kilo Auto Free", ("kilo", "kilo-free"), False, False, 200_000, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "openai/gpt-5.4", "GPT-5.4 (Kilo)", ("kilo-gpt5",), True, True, 1_000_000, 100_000),
        CatalogModel(provider.id, provider.api, provider.label, "openai/gpt-5.1", "GPT-5.1 (Kilo)", ("kilo-gpt5.1",), False, True, 1_047_576, 32_768),
        CatalogModel(provider.id, provider.api, provider.label, "anthropic/claude-opus-4-7", "Claude Opus 4.7 (Kilo)", ("kilo-opus",), True, True, 200_000, 32_000),
        CatalogModel(provider.id, provider.api, provider.label, "anthropic/claude-sonnet-4-6", "Claude Sonnet 4.6 (Kilo)", ("kilo-sonnet",), False, True, 200_000, 16_000),
        CatalogModel(provider.id, provider.api, provider.label, "xai/grok-3", "Grok 3 (Kilo)", ("kilo-grok",), True, False, 131_072, 131_072),
        CatalogModel(provider.id, provider.api, provider.label, "google/gemini-2.5-pro", "Gemini 2.5 Pro (Kilo)", ("kilo-gemini",), True, True, 1_048_576, 65_536),
        CatalogModel(provider.id, provider.api, provider.label, "deepseek/deepseek-r1", "DeepSeek R1 (Kilo)", ("kilo-r1",), True, False, 163_840, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "meta-llama/llama-4-maverick", "Llama 4 Maverick (Kilo)", ("kilo-llama4",), False, True, 524_288, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "free/giga-potato", "Giga Potato (free)", ("giga-potato-free",), False, False, 131_072, 8_192),
    )
    return provider, models


def _opencode_provider() -> tuple[ProviderConfig, tuple[CatalogModel, ...]] | None:
    if not _env("OPENCODE_API_KEY"):
        return None
    provider = ProviderConfig(
        "opencode",
        "openai",
        "opencode",
        ("OPENCODE_API_KEY",),
        base_url=_env("OPENCODE_BASE_URL") or "https://api.opencode.ai/v1",
    )
    models = (
        CatalogModel(provider.id, provider.api, provider.label, "gpt-5.4-codex", "GPT-5.4 Codex (OpenCode)", ("opencode", "opencode-codex"), True, True, 1_000_000, 100_000),
        CatalogModel(provider.id, provider.api, provider.label, "gpt-5.3-codex", "GPT-5.3 Codex (OpenCode)", ("opencode-5.3",), True, True, 1_000_000, 100_000),
        CatalogModel(provider.id, provider.api, provider.label, "claude-opus-4-7", "Claude Opus 4.7 (OpenCode)", ("opencode-opus",), True, True, 200_000, 32_000),
        CatalogModel(provider.id, provider.api, provider.label, "claude-sonnet-4-6", "Claude Sonnet 4.6 (OpenCode)", ("opencode-sonnet",), False, True, 200_000, 16_000),
        CatalogModel(provider.id, provider.api, provider.label, "gpt-5.1", "GPT-5.1 (OpenCode)", ("opencode-5.1",), False, True, 1_047_576, 32_768),
        CatalogModel(provider.id, provider.api, provider.label, "gemini-2.5-pro", "Gemini 2.5 Pro (OpenCode)", ("opencode-gemini",), True, True, 1_048_576, 65_536),
        CatalogModel(provider.id, provider.api, provider.label, "deepseek-r1", "DeepSeek R1 (OpenCode)", ("opencode-r1",), True, False, 163_840, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "o3", "o3 (OpenCode)", ("opencode-o3",), True, True, 200_000, 100_000),
    )
    return provider, models


def _opencode_zen_provider() -> tuple[ProviderConfig, tuple[CatalogModel, ...]] | None:
    if not _env("OPENCODE_ZEN_API_KEY"):
        return None
    provider = ProviderConfig(
        "opencode-zen",
        "openai",
        "opencode zen",
        ("OPENCODE_ZEN_API_KEY",),
        base_url=_env("OPENCODE_ZEN_BASE_URL") or "https://opencode.ai/zen/v1",
    )
    models = (
        CatalogModel(provider.id, provider.api, provider.label, "big-pickle", "Big Pickle (Zen)", ("zen-pickle",), True, True, 1_000_000, 100_000),
        CatalogModel(provider.id, provider.api, provider.label, "gpt-5.4", "GPT-5.4 (Zen)", ("zen-5.4",), True, True, 1_000_000, 100_000),
        CatalogModel(provider.id, provider.api, provider.label, "gpt-5.4-nano", "GPT-5.4 Nano (Zen)", ("zen-nano",), False, True, 1_000_000, 16_384),
        CatalogModel(provider.id, provider.api, provider.label, "gpt-5.3-codex", "GPT-5.3 Codex (Zen)", ("opencode-zen", "zen"), True, True, 1_000_000, 100_000),
        CatalogModel(provider.id, provider.api, provider.label, "gpt-5.2-codex", "GPT-5.2 Codex (Zen)", ("zen-5.2-codex",), True, True, 1_000_000, 100_000),
        CatalogModel(provider.id, provider.api, provider.label, "gpt-5-codex", "GPT-5 Codex (Zen)", ("zen-5-codex",), True, True, 1_000_000, 100_000),
        CatalogModel(provider.id, provider.api, provider.label, "gpt-5.1-codex-max", "GPT-5.1 Codex Max (Zen)", ("zen-codex-max",), True, True, 1_000_000, 100_000),
        CatalogModel(provider.id, provider.api, provider.label, "claude-opus-4-7", "Claude Opus 4.7 (Zen)", ("zen-opus",), True, True, 200_000, 32_000),
        CatalogModel(provider.id, provider.api, provider.label, "claude-opus-4-6", "Claude Opus 4.6 (Zen)", ("zen-opus-4-6",), True, True, 200_000, 32_000),
        CatalogModel(provider.id, provider.api, provider.label, "claude-sonnet-4-6", "Claude Sonnet 4.6 (Zen)", ("zen-sonnet",), False, True, 200_000, 16_000),
        CatalogModel(provider.id, provider.api, provider.label, "claude-haiku-4-5", "Claude Haiku 4.5 (Zen)", ("zen-haiku",), False, True, 200_000, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "gemini-3.1-pro-preview", "Gemini 3.1 Pro (Zen)", ("zen-gemini-3",), True, True, 1_048_576, 65_536),
        CatalogModel(provider.id, provider.api, provider.label, "gemini-3-flash", "Gemini 3 Flash (Zen)", ("zen-gemini-flash",), False, True, 1_048_576, 32_768),
        CatalogModel(provider.id, provider.api, provider.label, "glm-5.1", "GLM-5.1 (Zen)", ("zen-glm-5",), False, False, 128_000, 4_096),
        CatalogModel(provider.id, provider.api, provider.label, "kimi-k2.5", "Kimi K2.5 (Zen)", ("zen-kimi",), False, False, 200_000, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "minimax-m2.5-free", "MiniMax M2.5 (free)", ("zen-free", "minimax-free"), False, False, 1_000_000, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "nemotron-3-super-free", "Nemotron 3 Super (free)", ("nemotron-free",), True, False, 131_072, 16_384),
        CatalogModel(provider.id, provider.api, provider.label, "qwen3.6-plus-free", "Qwen 3.6 Plus (free)", ("qwen-zen-free",), False, False, 131_072, 8_192),
    )
    return provider, models


def _anthropic_provider() -> tuple[ProviderConfig, tuple[CatalogModel, ...]] | None:
    if not _env("ANTHROPIC_API_KEY"):
        return None
    provider = ProviderConfig("anthropic", "anthropic", "anthropic", ("ANTHROPIC_API_KEY",))
    models = (
        CatalogModel(provider.id, provider.api, provider.label, "claude-opus-4-7", "Claude Opus 4.7", ("claude-opus-4-7-direct",), True, True, 200_000, 32_000),
        CatalogModel(provider.id, provider.api, provider.label, "claude-sonnet-4-6", "Claude Sonnet 4.6", ("claude-sonnet-4-6-direct", "claude-direct"), False, True, 200_000, 16_000),
        CatalogModel(provider.id, provider.api, provider.label, "claude-haiku-4-5-20251001", "Claude Haiku 4.5 (Direct)", ("claude-haiku-direct",), False, True, 200_000, 8_192),
    )
    return provider, models


def _gemini_provider() -> tuple[ProviderConfig, tuple[CatalogModel, ...]] | None:
    if not _env("GEMINI_API_KEY"):
        return None
    provider = ProviderConfig("gemini", "gemini", "gemini", ("GEMINI_API_KEY",))
    models = (
        CatalogModel(provider.id, provider.api, provider.label, "gemini-2.5-pro",       "Gemini 2.5 Pro",        ("gemini-pro", "gemini"),  True,  True, 1_048_576, 65_536),
        CatalogModel(provider.id, provider.api, provider.label, "gemini-2.5-flash",     "Gemini 2.5 Flash",      ("gemini-2.5-flash",),     True,  True, 1_048_576, 65_536),
        CatalogModel(provider.id, provider.api, provider.label, "gemini-2.5-flash-lite","Gemini 2.5 Flash Lite", ("gemini-2.5-flash-lite",),False,  True, 1_048_576, 65_536),
        CatalogModel(provider.id, provider.api, provider.label, "gemini-2.0-flash",     "Gemini 2.0 Flash",      ("gemini-flash",),         False, True, 1_048_576,  8_192),
        CatalogModel(provider.id, provider.api, provider.label, "gemini-2.0-flash-lite","Gemini 2.0 Flash Lite", ("gemini-flash-lite",),    False, True, 1_048_576,  8_192),
        CatalogModel(provider.id, provider.api, provider.label, "gemini-1.5-pro",       "Gemini 1.5 Pro",        ("gemini-1.5-pro",),       False, True, 2_000_000,  8_192),
        CatalogModel(provider.id, provider.api, provider.label, "gemini-1.5-flash",     "Gemini 1.5 Flash",      ("gemini-1.5-flash",),     False, True, 1_000_000,  8_192),
        CatalogModel(provider.id, provider.api, provider.label, "gemini-1.5-flash-8b",  "Gemini 1.5 Flash 8B",   ("gemini-1.5-flash-8b",),  False, True, 1_000_000,  8_192),
    )
    return provider, models


def _bedrock_provider() -> tuple[ProviderConfig, tuple[CatalogModel, ...]] | None:
    region = _env("AWS_REGION") or _env("AWS_DEFAULT_REGION") or _env("BEDROCK_REGION")
    has_auth = any(
        _env(key)
        for key in (
            "AWS_BEARER_TOKEN_BEDROCK",
            "AWS_ACCESS_KEY_ID",
            "AWS_PROFILE",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
            "BEDROCK_ASSUME_ROLE_ARN",
        )
    )
    if not region or not has_auth:
        return None

    provider = ProviderConfig(
        id=BEDROCK_PROVIDER_ID,
        api="bedrock",
        label=BEDROCK_PROVIDER_ID,
        env_keys=(),
        options={
            "region": region,
            "profile": _env("AWS_PROFILE"),
            "endpoint": _env("BEDROCK_ENDPOINT_URL"),
            "bearer_token": _env("AWS_BEARER_TOKEN_BEDROCK"),
            "role_arn": _env("BEDROCK_ASSUME_ROLE_ARN"),
            "session_name": _env("BEDROCK_ASSUME_ROLE_SESSION_NAME") or "ai-router-bedrock",
        },
    )

    configured = [item.strip() for item in _env("BEDROCK_MODELS").split(",") if item.strip()]
    defaults = configured or [
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "anthropic.claude-3-5-haiku-20241022-v1:0",
        "anthropic.claude-3-7-sonnet-20250219-v1:0",
    ]
    aliases = {
        "anthropic.claude-3-5-sonnet-20241022-v2:0": ("bedrock-sonnet",),
        "anthropic.claude-3-5-haiku-20241022-v1:0": ("bedrock-haiku",),
        "anthropic.claude-3-7-sonnet-20250219-v1:0": ("bedrock-sonnet-thinking",),
    }

    models = []
    for model_id in defaults:
        canonical = _maybe_prefix_bedrock_model_id(model_id, region)
        display_name = f"{canonical} (Amazon Bedrock)"
        models.append(
            CatalogModel(
                provider.id,
                provider.api,
                provider.label,
                canonical,
                display_name,
                aliases.get(model_id, ()) + aliases.get(canonical, ()),
                reasoning="sonnet" in model_id,
                vision="claude" in model_id,
                context_window=200_000,
                max_tokens=8_192,
            )
        )
    return provider, tuple(models)


def _provider_specs() -> list[tuple[ProviderConfig, tuple[CatalogModel, ...]]]:
    providers = []
    for builder in (
        _auto_routing_provider,
        _anthropic_provider,
        _openai_provider,
        _gemini_provider,
        _mistral_provider,
        _deepseek_provider,
        _xai_provider,
        _together_provider,
        _perplexity_provider,
        _fireworks_provider,
        _cohere_provider,
        _copilot_provider,
        _groq_provider,
        _cerebras_provider,
        _openrouter_provider,
        _zai_provider,
        _kilo_provider,
        _opencode_provider,
        _opencode_zen_provider,
        _bedrock_provider,
    ):
        spec = builder()
        if spec:
            providers.append(spec)
    return providers


def _resolve_api_key(provider: ProviderConfig) -> str:
    for key in provider.env_keys:
        value = _env(key)
        if value:
            return value
    return ""


def build_registry() -> RegistryState:
    models_dev = _index_models_dev(_load_models_dev())
    state = RegistryState()

    for provider, catalog in _provider_specs():
        if not _provider_allowed(provider.id):
            continue
        api_key = _resolve_api_key(provider)

        for raw_model in catalog:
            if not _model_allowed(raw_model.model_id):
                continue
            if not _provider_model_allowed(provider.id, raw_model.model_id):
                continue
            model = _enrich_model(raw_model, models_dev)
            entry = ModelEntry(
                provider=provider.api,
                provider_id=provider.id,
                provider_label=provider.label,
                model_id=model.model_id,
                api_key=api_key,
                name=model.name,
                reasoning=model.reasoning,
                vision=model.vision,
                context_window=model.context_window,
                max_tokens=model.max_tokens,
                base_url=provider.base_url,
                extra_headers=dict(provider.extra_headers),
                aliases=model.aliases,
                options=dict(provider.options),
            )
            state.by_canonical_id[model.model_id] = entry
            keys = [model.model_id, f"{provider.id}/{model.model_id}", f"{provider.api}/{model.model_id}", *model.aliases]
            for key in keys:
                normalised = _normalise_key(key)
                if normalised:
                    state.aliases[normalised] = model.model_id

    return state


REGISTRY = RegistryState()


def init(env_path: str | None = None) -> None:
    global REGISTRY
    if env_path:
        _load_dotenv(env_path)
    REGISTRY = build_registry()


def get(model_name: str) -> ModelEntry | None:
    if not model_name:
        return None
    canonical = REGISTRY.aliases.get(_normalise_key(model_name), model_name)
    return REGISTRY.by_canonical_id.get(canonical) or REGISTRY.by_canonical_id.get(_normalise_key(canonical))


def list_models() -> list[dict]:
    out: list[dict] = []
    for model_id, entry in sorted(REGISTRY.by_canonical_id.items(), key=lambda item: item[0]):
        out.append(
            {
                "id": model_id,
                "object": "model",
                "owned_by": entry.provider_label or entry.provider_id or entry.provider,
                "provider_api": entry.provider,
                "provider_id": entry.provider_id or entry.provider,
                "name": entry.name,
                "reasoning": entry.reasoning,
                "vision": entry.vision,
                "context_window": entry.context_window,
                "max_tokens": entry.max_tokens,
                "aliases": list(entry.aliases),
                "primary": True,
            }
        )
    return out


def _load_dotenv(path: str) -> None:
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
