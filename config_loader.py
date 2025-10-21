import os
from dataclasses import dataclass
from typing import Optional


def _load_toml(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
            import tomli as tomllib  # type: ignore
        with open(path, 'rb') as f:
            return tomllib.load(f)
    except Exception:
        return {}


@dataclass
class AppConfig:
    proxy_url: Optional[str]
    openai_base_url: Optional[str]
    openai_api_key: Optional[str]
    disable_ssl: bool
    # ALM Octane integration
    octane_base_url: Optional[str]
    octane_client_id: Optional[str]
    octane_client_secret: Optional[str]
    # Confluence integration
    confluence_base_url: Optional[str]
    confluence_access_token: Optional[str]
    mode: str
    model_name: str
    embedding_model_name: Optional[str]
    secret_key: Optional[str]
    # Local TTS (Coqui TTS)
    coqui_tts_model: Optional[str]
    coqui_tts_device: Optional[str]
    coqui_tts_language: Optional[str]


def load_config() -> AppConfig:
    cfg = _load_toml(os.path.join(os.getcwd(), 'config.toml'))

    def get_cfg(key: str, env: str, default=None):
        # Environment variables can override config, but are not required.
        if env in os.environ and os.environ[env] != "":
            return os.environ[env]
        v = cfg.get(key) if isinstance(cfg, dict) else None
        return v if v not in (None, "") else default

    proxy_url = get_cfg('proxy_url', 'PROXY_URL', None)
    openai_base_url = get_cfg('openai_base_url', 'OPENAI_BASE_URL', None)
    openai_api_key = get_cfg('openai_api_key', 'OPENAI_API_KEY', None)
    # Octane
    octane_base_url = get_cfg('octane_base_url', 'OCTANE_BASE_URL', None)
    octane_client_id = get_cfg('octane_client_id', 'OCTANE_CLIENT_ID', None)
    octane_client_secret = get_cfg('octane_client_secret', 'OCTANE_CLIENT_SECRET', None)
    confluence_base_url = get_cfg('confluence_base_url', 'CONFLUENCE_BASE_URL', None)
    confluence_access_token = get_cfg('confluence_access_token', 'CONFLUENCE_ACCESS_TOKEN', None)
    disable_ssl_raw = get_cfg('disable_ssl', 'DISABLE_SSL', False)
    mode = str(get_cfg('mode', 'MODE', 'chat')).lower()
    if mode not in ("chat",):
        mode = "chat"
    model_name = get_cfg('model_name', 'MODEL_NAME', 'gpt-4o-mini')
    embedding_model_name = get_cfg('embedding_model_name', 'EMBEDDING_MODEL_NAME', 'text-embedding-3-small')
    secret_key = get_cfg('secret_key', 'SECRET_KEY', None)
    coqui_tts_model = get_cfg('coqui_tts_model', 'COQUI_TTS_MODEL', None)
    coqui_tts_device = get_cfg('coqui_tts_device', 'COQUI_TTS_DEVICE', None)
    coqui_tts_language = get_cfg('coqui_tts_language', 'COQUI_TTS_LANGUAGE', None)

    disable_ssl = False
    if isinstance(disable_ssl_raw, str):
        disable_ssl = disable_ssl_raw.lower() in ("1", "true", "yes", "on")
    elif isinstance(disable_ssl_raw, bool):
        disable_ssl = disable_ssl_raw

    return AppConfig(
        proxy_url=proxy_url or None,
        openai_base_url=openai_base_url or None,
        openai_api_key=openai_api_key or None,
        disable_ssl=disable_ssl,
        octane_base_url=octane_base_url or None,
        octane_client_id=octane_client_id or None,
        octane_client_secret=octane_client_secret or None,
        confluence_base_url=confluence_base_url or None,
        confluence_access_token=confluence_access_token or None,
        mode=mode,
        model_name=str(model_name),
        embedding_model_name=str(embedding_model_name) if embedding_model_name else None,
        secret_key=secret_key,
        coqui_tts_model=coqui_tts_model or None,
        coqui_tts_device=coqui_tts_device or None,
        coqui_tts_language=coqui_tts_language or None,
    )
