"""
Core bridge logic for managing provider state and configuration.
Supports both single model and model-map modes.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from ..providers.base import ProviderPlugin
from ..providers.openai_compat import OpenAICompatPlugin

logger = logging.getLogger("prism.bridge")


# Built-in model aliases for common providers
BUILTIN_ALIASES = {
    "https://api.mistral.ai/v1": {
        "claude-opus-4-7": "mistral-large-latest",
        "claude-opus-4-5": "mistral-large-latest",
        "claude-sonnet-4-6": "codestral-latest",
        "claude-sonnet-4-5": "codestral-latest",
        "claude-haiku-4-5": "ministral-8b-latest",
        "claude-haiku-3-5": "ministral-3b-latest",
    },
    "https://api.groq.com/openai/v1": {
        "claude-opus-4-7": "llama-3.2-90b-vision-preview",
        "claude-opus-4-5": "llama-3.2-90b-vision-preview",
        "claude-sonnet-4-6": "llama-3.1-70b-versatile",
        "claude-sonnet-4-5": "llama-3.1-70b-versatile",
        "claude-haiku-4-5": "llama-3.1-8b-instant",
        "claude-haiku-3-5": "llama-3.1-8b-instant",
    },
    "https://integrate.api.nvidia.com/v1": {
        "claude-opus-4-7": "meta/llama-3.1-405b-instruct",
        "claude-sonnet-4-6": "meta/llama-3.1-70b-instruct",
        "claude-haiku-4-5": "meta/llama-3.1-8b-instruct",
    },
    "https://api.together.xyz/v1": {
        "claude-opus-4-7": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "claude-sonnet-4-6": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "claude-haiku-4-5": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    },
    "https://openrouter.ai/api/v1": {
        "claude-opus-4-7": "anthropic/claude-opus-4.1",
        "claude-opus-4-5": "anthropic/claude-opus-4.1",
        "claude-sonnet-4-6": "anthropic/claude-sonnet-4",
        "claude-sonnet-4-5": "anthropic/claude-sonnet-4",
        "claude-haiku-4-5": "anthropic/claude-3.5-haiku",
        "claude-haiku-3-5": "anthropic/claude-3.5-haiku",
    },
}


@dataclass
class BridgeConfig:
    """Configuration for the bridge."""
    provider_url: str = ""
    api_key: str | None = None
    api_keys: list[str] = field(default_factory=list)
    model: str | None = None
    model_map: dict[str, str] = field(default_factory=dict)
    fallback_model: str | None = None
    key_index: int = 0
    exhausted_keys: set[int] = field(default_factory=set)
    plugin: ProviderPlugin | None = None


class Bridge:
    """Manages live state of both ends of the proxy."""

    def __init__(self):
        self.config = BridgeConfig()
        self._plugin: ProviderPlugin | None = None
        self._ready = False
        self._client_format = "unknown"
        self._client_tool = "unknown"

    def configure(
        self,
        provider_url: str,
        api_key: str | None = None,
        api_keys: list[str] | None = None,
        model: str | None = None,
        model_map: dict[str, str] | None = None,
        fallback_model: str | None = None,
    ) -> None:
        """Configure the bridge with provider settings."""
        self.config.provider_url = provider_url.rstrip("/")
        self.config.api_key = api_key
        self.config.api_keys = api_keys or []
        self.config.model = model
        self.config.model_map = model_map or {}
        self.config.fallback_model = fallback_model
        self.config.key_index = 0
        self.config.exhausted_keys = set()
        self._ready = True

    @property
    def plugin(self) -> ProviderPlugin:
        if self._plugin is None:
            self._plugin = OpenAICompatPlugin()
        return self._plugin

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def client_format(self) -> str:
        return self._client_format

    def get_current_api_key(self) -> str | None:
        """Get current API key, cycling through available keys."""
        if self.config.api_keys:
            if self.config.exhausted_keys and len(self.config.exhausted_keys) >= len(self.config.api_keys):
                self.config.exhausted_keys.clear()
                self.config.key_index = 0
            return self.config.api_keys[self.config.key_index]
        return self.config.api_key

    def mark_key_exhausted(self, index: int) -> None:
        """Mark a key as exhausted (rate limited or quota exceeded)."""
        if self.config.api_keys:
            self.config.exhausted_keys.add(index)

    def advance_key(self) -> None:
        """Advance to next available key in round-robin."""
        if not self.config.api_keys:
            return
        original_idx = self.config.key_index
        attempts = 0
        while attempts < len(self.config.api_keys):
            self.config.key_index = (self.config.key_index + 1) % len(self.config.api_keys)
            if self.config.key_index not in self.config.exhausted_keys:
                return
            attempts += 1
        self.config.exhausted_keys.clear()
        self.config.key_index = (original_idx + 1) % len(self.config.api_keys)

    def resolve_model(self, requested_model: str | None) -> str | None:
        """Resolve which provider model to use for a frontend model request."""
        if self.config.model_map:
            if requested_model and requested_model in self.config.model_map:
                return self.config.model_map[requested_model]
            return self.config.fallback_model or self.config.model
        return self.config.model

    def is_configured(self) -> bool:
        return self._ready and self.config.provider_url != ""

    def status(self) -> dict[str, Any]:
        return {
            "ready": self._ready,
            "mode": "model-map" if self.config.model_map else "single",
            "model": self.config.model,
            "model_map": self.config.model_map,
            "fallback_model": self.config.fallback_model,
            "provider_url": self.config.provider_url,
            "client_format": self._client_format,
            "client_tool": self._client_tool,
        }


# Global bridge instance
_bridge = Bridge()


def get_bridge() -> Bridge:
    return _bridge
