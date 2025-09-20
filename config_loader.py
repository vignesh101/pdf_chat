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
    model_name: str
    secret_key: Optional[str]


def load_config() -> AppConfig:
    cfg = _load_toml(os.path.join(os.getcwd(), 'config.toml'))

    def get_cfg(key: str, env: str, default=None):
        if env in os.environ and os.environ[env] != "":
            return os.environ[env]
        v = cfg.get(key) if isinstance(cfg, dict) else None
        return v if v not in (None, "") else default

    proxy_url = get_cfg('proxy_url', 'PROXY_URL', None)
    openai_base_url = get_cfg('openai_base_url', 'OPENAI_BASE_URL', None)
    openai_api_key = get_cfg('openai_api_key', 'OPENAI_API_KEY', None)
    disable_ssl_raw = get_cfg('disable_ssl', 'DISABLE_SSL', False)
    model_name = get_cfg('model_name', 'MODEL_NAME', 'gpt-4o-mini')
    secret_key = os.environ.get('SECRET_KEY')

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
        model_name=str(model_name),
        secret_key=secret_key,
    )

