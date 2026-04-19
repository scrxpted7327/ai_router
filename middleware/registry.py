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
        CatalogModel(provider.id, provider.api, provider.label, "claude-opus-4-5", "Claude Opus 4.5 (Copilot)", ("claude-opus", "opus"), True, True, 200_000, 32_000),
        CatalogModel(provider.id, provider.api, provider.label, "claude-sonnet-4-5", "Claude Sonnet 4.5 (Copilot)", ("claude-sonnet", "claude", "sonnet"), True, True, 200_000, 16_000),
        CatalogModel(provider.id, provider.api, provider.label, "claude-haiku-4-5-20251001", "Claude Haiku 4.5 (Copilot)", ("claude-haiku", "haiku"), False, True, 200_000, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet (Copilot)", ("claude-3-5-sonnet",), False, True, 200_000, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "claude-3-5-haiku-20241022", "Claude 3.5 Haiku (Copilot)", ("claude-3-5-haiku",), False, True, 200_000, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "gpt-4o", "GitHub Copilot (GPT-4o)", ("github-copilot", "copilot"), False, True, 128_000, 16_384),
    )
    return provider, models


def _groq_provider() -> tuple[ProviderConfig, tuple[CatalogModel, ...]] | None:
    if not _env("GROQ_API_KEY"):
        return None
    provider = ProviderConfig("groq", "openai", "groq", ("GROQ_API_KEY",), base_url="https://api.groq.com/openai/v1")
    models = (
        CatalogModel(provider.id, provider.api, provider.label, "llama-3.3-70b-versatile", "Llama 3.3 70B (Groq)", ("llama-3.3-70b", "llama", "groq"), False, False, 128_000, 32_768),
        CatalogModel(provider.id, provider.api, provider.label, "llama-3.1-8b-instant", "Llama 3.1 8B Instant (Groq)", ("llama-fast",), False, False, 128_000, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "deepseek-r1-distill-llama-70b", "DeepSeek R1 Distill (Groq)", ("deepseek-r1-groq",), True, False, 128_000, 16_384),
    )
    return provider, models


def _cerebras_provider() -> tuple[ProviderConfig, tuple[CatalogModel, ...]] | None:
    if not _env("CEREBRAS_API_KEY"):
        return None
    provider = ProviderConfig("cerebras", "openai", "cerebras", ("CEREBRAS_API_KEY",), base_url="https://api.cerebras.ai/v1")
    models = (
        CatalogModel(provider.id, provider.api, provider.label, "llama-3.3-70b", "Llama 3.3 70B (Cerebras)", ("llama-3.3-70b-cerebras", "cerebras", "cerebras-fast"), False, False, 128_000, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "qwen-3-32b", "Qwen3 32B (Cerebras)", ("qwen-3-32b-cerebras", "qwen-cerebras"), True, False, 32_000, 16_000),
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
        CatalogModel(provider.id, provider.api, provider.label, "auto", "Auto (OpenRouter)", ("openrouter",), False, False, 200_000, 16_384),
        CatalogModel(provider.id, provider.api, provider.label, "deepseek/deepseek-chat", "DeepSeek V3 (OpenRouter)", ("deepseek", "deepseek-v3"), False, False, 163_840, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "deepseek/deepseek-r1", "DeepSeek R1 (OpenRouter)", ("deepseek-r1",), True, False, 163_840, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "qwen/qwen-2.5-72b-instruct", "Qwen 2.5 72B (OpenRouter)", ("qwen", "qwen-2.5"), False, False, 131_072, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "mistralai/mistral-7b-instruct:free", "Mistral 7B (free)", ("free", "free-mistral"), False, False, 32_768, 4_096),
        CatalogModel(provider.id, provider.api, provider.label, "meta-llama/llama-3.1-8b-instruct:free", "Llama 3.1 8B (free)", ("free-llama",), False, False, 131_072, 8_192),
    )
    return provider, models


def _zai_provider() -> tuple[ProviderConfig, tuple[CatalogModel, ...]] | None:
    if not _env("ZAI_API_KEY"):
        return None
    provider = ProviderConfig("zai", "openai", "zai", ("ZAI_API_KEY",), base_url="https://open.bigmodel.cn/api/paas/v4")
    models = (
        CatalogModel(provider.id, provider.api, provider.label, "glm-4-plus", "GLM-4 Plus (ZAI)", ("zai",), False, True, 128_000, 4_096),
    )
    return provider, models


def _kilo_provider() -> tuple[ProviderConfig, tuple[CatalogModel, ...]] | None:
    if not _env("KILO_API_KEY"):
        return None
    provider = ProviderConfig("kilo", "openai", "kilo", ("KILO_API_KEY",), base_url="https://api.kilo.ai/v1")
    models = (
        CatalogModel(provider.id, provider.api, provider.label, "kilo-default", "Kilo Default", ("kilo",), False, False, 200_000, 8_192),
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
        CatalogModel(provider.id, provider.api, provider.label, "opencode-default", "OpenCode Default", ("opencode",), False, False, 200_000, 8_192),
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
        base_url=_env("OPENCODE_ZEN_BASE_URL") or "https://api.minimaxi.chat/v1",
    )
    models = (
        CatalogModel(provider.id, provider.api, provider.label, "MiniMax-Text-01", "MiniMax Text-01 (OpenCode Zen)", ("opencode-zen", "minimax"), False, False, 1_000_000, 8_192),
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
        CatalogModel(provider.id, provider.api, provider.label, "gemini-2.5-pro", "Gemini 2.5 Pro", ("gemini-pro", "gemini"), True, True, 1_048_576, 65_536),
        CatalogModel(provider.id, provider.api, provider.label, "gemini-2.0-flash", "Gemini 2.0 Flash", ("gemini-flash",), False, True, 1_048_576, 8_192),
        CatalogModel(provider.id, provider.api, provider.label, "gemini-2.0-flash-lite", "Gemini 2.0 Flash Lite", ("gemini-flash-lite",), False, True, 1_048_576, 8_192),
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
        _anthropic_provider,
        _gemini_provider,
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
