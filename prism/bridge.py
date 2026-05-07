"""
Bridge — holds live state of both ends.
Supports both single model and model-map modes.
"""

from dataclasses import dataclass, field
from .probe.client   import ClientSchema
from .probe.provider import ProviderSchema


@dataclass
class Bridge:
    # Provider config
    provider:        ProviderSchema | None = None
    api_key:         str | None            = None
    provider_format: str                   = "openai-compat"
    completion_url:  str                   = ""

    # Single model mode
    model:           str | None            = None

    # Model map mode: frontend_model → provider_model
    # e.g. {"claude-opus-4-7": "mistral-large-latest", "claude-sonnet-4-6": "mistral-small-latest"}
    model_map:       dict[str, str]        = field(default_factory=dict)

    # Fallback model when request model not in map
    fallback_model:  str | None            = None

    # Client
    client:          ClientSchema | None   = None
    ready:           bool                  = False

    def is_configured(self) -> bool:
        return (
            self.completion_url != ""
            and (self.model is not None or len(self.model_map) > 0)
        )

    def mark_ready(self):
        self.ready = True

    def resolve_model(self, requested_model: str | None) -> str | None:
        """
        Resolve which provider model to use for a given frontend model request.
        Priority: model_map → fallback_model → single model → None
        """
        if self.model_map:
            if requested_model and requested_model in self.model_map:
                return self.model_map[requested_model]
            # Not in map or None — use fallback
            return self.fallback_model or self.model

        # Single model mode
        return self.model

    def status(self) -> dict:
        return {
            "ready":           self.ready,
            "mode":            "model-map" if self.model_map else "single",
            "model":           self.model,
            "model_map":       self.model_map,
            "fallback_model":  self.fallback_model,
            "provider_format": self.provider_format,
            "completion_url":  self.completion_url,
            "client_format":   self.client.format    if self.client else "not learned yet",
            "client_tool":     self.client.tool_hint if self.client else "not learned yet",
        }


_bridge = Bridge()


def get_bridge() -> Bridge:
    return _bridge