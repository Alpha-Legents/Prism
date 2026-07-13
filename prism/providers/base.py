"""Provider plugin base class and registry."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Any


class ProviderPlugin(ABC):
    """Base class for provider-specific translation plugins."""

    name: str = "base"

    @abstractmethod
    def translate_request(self, body: dict, model_override: str | None = None) -> dict:
        """Translate an Anthropic-format request to provider format."""
        ...

    @abstractmethod
    def translate_response(self, raw: dict) -> dict:
        """Translate a provider response to Anthropic format."""
        ...

    @abstractmethod
    def translate_stream_chunk(self, chunk: dict) -> list[dict]:
        """Translate a single provider stream chunk to Anthropic SSE events."""
        ...

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming."""
        ...

    @property
    @abstractmethod
    def supports_tool_use(self) -> bool:
        """Whether this provider supports tool use."""
        ...

    @property
    @abstractmethod
    def supports_thinking(self) -> bool:
        """Whether this provider supports thinking blocks."""
        ...
