from __future__ import annotations

import httpx
from typing import Optional
from openai import OpenAI
from config_loader import AppConfig


def build_openai_client(cfg: AppConfig) -> OpenAI:
    # Create a dedicated HTTPX client to support proxies and SSL toggle
    http_client = httpx.Client(
        timeout=httpx.Timeout(60.0, connect=30.0, read=60.0, write=60.0),
        proxy=cfg.proxy_url if cfg.proxy_url else None,
        verify=False if cfg.disable_ssl else True,
    )

    client = OpenAI(
        api_key=cfg.openai_api_key or None,
        base_url=cfg.openai_base_url or None,
        http_client=http_client,
    )
    return client

