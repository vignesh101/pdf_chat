from __future__ import annotations

import httpx
from typing import Optional
from openai import OpenAI
from openai import OpenAIError
from config_loader import AppConfig


def build_openai_client(cfg: AppConfig) -> Optional[OpenAI]:
    # Create a dedicated HTTPX client to support proxies and SSL toggle
    http_client = build_httpx_client(cfg)

    try:
        client = OpenAI(
            api_key=cfg.openai_api_key or None,
            base_url=cfg.openai_base_url or None,
            http_client=http_client,
        )
        # The OpenAI client raises if api_key is missing; catch and return None
        return client
    except OpenAIError:
        return None


def build_httpx_client(cfg: AppConfig) -> httpx.Client:
    """Build a plain HTTPX client honoring proxy and SSL verify settings."""
    return httpx.Client(
        timeout=httpx.Timeout(60.0, connect=30.0, read=60.0, write=60.0),
        proxy=cfg.proxy_url if cfg.proxy_url else None,
        verify=False if cfg.disable_ssl else True,
        headers={
            # Modest default UA to avoid overly suspicious default client idents
            'User-Agent': 'document-chat/1.0 (+https://localhost)'
        },
    )
