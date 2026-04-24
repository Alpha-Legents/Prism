"""
Provider probe.
Hits a provider endpoint, discovers available models, learns response shape.
Zero assumptions about which provider it is.
"""

import httpx
import logging
from dataclasses import dataclass, field
from typing import Any
from ..slots import detect_format

logger = logging.getLogger("prism.probe.provider")

# Minimal probe payload — works for both OpenAI-compat and Anthropic
PROBE_PAYLOAD_OPENAI = {
    "model": "__probe__",      # will fail but response shape is what matters
    "max_tokens": 1,
    "messages": [{"role": "user", "content": "hi"}],
}

PROBE_PAYLOAD_ANTHROPIC = {
    "model": "__probe__",
    "max_tokens": 1,
    "messages": [{"role": "user", "content": "hi"}],
}


@dataclass
class ProviderSchema:
    url: str
    format: str                        # 'openai-compat' | 'anthropic' | 'gemini'
    models: list[str] = field(default_factory=list)
    sample_response: dict = field(default_factory=dict)
    headers_seen: dict = field(default_factory=dict)
    reachable: bool = False
    api_key: str | None = None
    completion_url: str = ""

    def __repr__(self):
        return f"<Provider format={self.format} models={len(self.models)} reachable={self.reachable}>"


async def probe_provider(url: str, api_key: str | None = None) -> ProviderSchema:
    """
    Probe a provider endpoint:
    1. Try GET /models to discover available models
    2. Fire a minimal completion request to learn response shape
    """
    schema = ProviderSchema(url=url.rstrip("/"), format="unknown")

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"]      = f"Bearer {api_key}"
        headers["x-api-key"]          = api_key
        headers["anthropic-version"]  = "2023-06-01"

    async with httpx.AsyncClient(timeout=20) as client:

        # ── Step 1: Discover models ───────────────────────────────────────────
        for models_path in ["/v1/models", "/models"]:
            try:
                base = _base_url(url)
                r = await client.get(base + models_path, headers=headers)
                if r.status_code == 200:
                    data = r.json()
                    schema.models = _extract_models(data)
                    logger.info(f"Found {len(schema.models)} models at {models_path}")
                    break
            except Exception as e:
                logger.debug(f"Model fetch failed ({models_path}): {e}")

        # ── Step 2: Learn response shape ──────────────────────────────────────
        # Try OpenAI-compat first, then Anthropic
        completion_url = _completion_url(url)

        for payload in [PROBE_PAYLOAD_OPENAI, PROBE_PAYLOAD_ANTHROPIC]:
            try:
                r = await client.post(completion_url, json=payload, headers=headers)
                schema.headers_seen = dict(r.headers)
                schema.reachable    = True

                try:
                    body = r.json()
                    fmt  = detect_format(body)
                    if fmt != "unknown":
                        schema.format          = fmt
                        schema.sample_response = body
                        logger.info(f"Provider format detected: {fmt}")
                        break
                    # Even error responses reveal shape
                    if "error" in body:
                        # Infer from error structure
                        if "message" in body.get("error", {}):
                            schema.format = "openai-compat"
                        elif "type" in body:
                            schema.format = "anthropic"
                        schema.sample_response = body
                        break
                except Exception:
                    pass

            except Exception as e:
                logger.warning(f"Completion probe failed: {e}")

    return schema


def _base_url(url: str) -> str:
    """Strip path back to base — e.g. https://api.groq.com/openai/v1/chat/completions → https://api.groq.com/openai"""
    # If URL ends with known completion paths, strip them
    for suffix in ["/chat/completions", "/completions", "/messages"]:
        if url.endswith(suffix):
            return url[: -len(suffix)]
    # Otherwise strip last segment if it looks like an endpoint
    parts = url.rstrip("/").rsplit("/", 1)
    if parts[-1] in ("completions", "messages", "v1"):
        return parts[0]
    return url.rstrip("/")


def _completion_url(url: str) -> str:
    """Ensure URL points to a completion endpoint."""
    url = url.rstrip("/")
    if url.endswith(("/chat/completions", "/messages", "/completions")):
        return url
    # Try to guess
    if "anthropic" in url or "claude" in url:
        return url + "/messages"
    return url + "/chat/completions"


def _extract_models(data: Any) -> list[str]:
    """Pull model IDs out of whatever shape the /models endpoint returns."""
    if isinstance(data, list):
        return [m.get("id", str(m)) if isinstance(m, dict) else str(m) for m in data]
    if isinstance(data, dict):
        # OpenAI style: {"data": [{id: ...}]}
        if "data" in data:
            return [m.get("id", "") for m in data["data"] if isinstance(m, dict)]
        # Anthropic style: {"models": [{id: ...}]}
        if "models" in data:
            return [m.get("id", "") for m in data["models"] if isinstance(m, dict)]
    return []
