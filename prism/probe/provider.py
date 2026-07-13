"""Provider probe — hits provider, discovers models, learns response shape."""

import httpx
import logging
from dataclasses import dataclass, field
from typing import Any
from ..slots import detect_format

logger = logging.getLogger("prism.probe.provider")

PROBE_PAYLOAD = {
    "model": "__probe__",
    "max_tokens": 1,
    "messages": [{"role": "user", "content": "hi"}],
}


@dataclass
class ProviderSchema:
    url:             str
    format:          str
    models:          list[str] = field(default_factory=list)
    sample_response: dict      = field(default_factory=dict)
    headers_seen:    dict      = field(default_factory=dict)
    reachable:       bool      = False
    api_key:         str | None = None
    completion_url:  str        = ""

    def __repr__(self):
        return f"<Provider format={self.format} models={len(self.models)} reachable={self.reachable}>"


async def probe_provider(url: str, api_key: str | None = None) -> ProviderSchema:
    schema = ProviderSchema(url=url.rstrip("/"), format="unknown")

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"]     = f"Bearer {api_key}"
        headers["x-api-key"]         = api_key
        headers["anthropic-version"] = "2023-06-01"

    async with httpx.AsyncClient(timeout=20) as client:
        # Discover models
        for path in ["/v1/models", "/models"]:
            try:
                r = await client.get(base_url(url) + path, headers=headers)
                if r.status_code == 200:
                    schema.models = _extract_models(r.json())
                    break
            except Exception:
                pass

        # Learn response shape
        try:
            r = await client.post(completion_url(url), json=PROBE_PAYLOAD, headers=headers)
            schema.headers_seen = dict(r.headers)
            schema.reachable    = True
            try:
                body = r.json()
                fmt  = detect_format(body)
                if fmt != "unknown":
                    schema.format = fmt
                elif "error" in body:
                    schema.format = "openai-compat" if "message" in body.get("error", {}) else "anthropic"
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Completion probe failed: {e}")

    return schema


def base_url(url: str) -> str:
    for suffix in ["/chat/completions", "/completions", "/messages"]:
        if url.rstrip("/").endswith(suffix):
            return url.rstrip("/")[: -len(suffix)]
    return url.rstrip("/")


def completion_url(url: str) -> str:
    url = url.rstrip("/")
    if url.endswith(("/chat/completions", "/messages", "/completions")):
        return url
    if "anthropic" in url or "claude" in url:
        return url + "/messages"
    return url + "/chat/completions"


def _extract_models(data: Any) -> list[str]:
    if isinstance(data, list):
        return [m.get("id", str(m)) if isinstance(m, dict) else str(m) for m in data]
    if isinstance(data, dict):
        if "data" in data:
            return [m.get("id", "") for m in data["data"] if isinstance(m, dict)]
        if "models" in data:
            return [m.get("id", "") for m in data["models"] if isinstance(m, dict)]
    return []