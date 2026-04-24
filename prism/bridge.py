"""
Bridge.
Holds the live learned state of both ends.
The single source of truth for the proxy.
"""

from dataclasses import dataclass, field
from .probe.provider import ProviderSchema
from .probe.client   import ClientSchema


@dataclass
class Bridge:
    provider: ProviderSchema | None = None
    client:   ClientSchema   | None = None
    model:    str | None            = None   # Selected model override
    ready:    bool                  = False

    def is_configured(self) -> bool:
        return self.provider is not None and self.model is not None

    def mark_ready(self):
        self.ready = True

    def status(self) -> dict:
        return {
            "ready":            self.ready,
            "provider_format":  self.provider.format    if self.provider else None,
            "provider_models":  len(self.provider.models) if self.provider else 0,
            "client_format":    self.client.format      if self.client   else "not learned yet",
            "client_tool":      self.client.tool_hint   if self.client   else "not learned yet",
            "model":            self.model,
        }


# Global bridge instance — shared between proxy and TUI
_bridge = Bridge()


def get_bridge() -> Bridge:
    return _bridge
