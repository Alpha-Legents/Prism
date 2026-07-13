"""Smart caching layer with proper TTL and field-based deduplication."""

import time
import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger("prism.cache")

# Cache configuration
DEFAULT_TTL = 3.0  # seconds — short enough for interactive use


class CacheEntry:
    """A single cached response with metadata."""

    def __init__(self, response: Any, headers: dict, fingerprint: str):
        self.response = response
        self.headers = headers
        self.fingerprint = fingerprint
        self.timestamp = time.time()
        self.hit_count = 0

    def is_expired(self, ttl: float = DEFAULT_TTL) -> bool:
        return time.time() - self.timestamp > ttl


class ResponseCache:
    """Thread-safe(ish) response cache with per-key isolation."""

    def __init__(self, ttl: float = DEFAULT_TTL):
        self._cache: dict[str, CacheEntry] = {}
        self._ttl = ttl
        self._hits = 0
        self._misses = 0

    def _fingerprint(self, body: dict, stream: bool = False) -> str:
        """Generate a stable fingerprint including ALL request fields."""
        # Include all fields that affect output
        data = {
            "model": body.get("model", ""),
            "stream": stream,
            "messages": body.get("messages", []),
            "tools": body.get("tools", []),
            "temperature": body.get("temperature"),
            "max_tokens": body.get("max_tokens"),
            "top_p": body.get("top_p"),
            "system": body.get("system"),
        }
        raw = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, body: dict, stream: bool = False) -> tuple[Any, dict] | None:
        """Get cached response if available and not expired."""
        fp = self._fingerprint(body, stream)

        # Clean expired entries periodically
        self._cleanup()

        if fp in self._cache:
            entry = self._cache[fp]
            if not entry.is_expired(self._ttl):
                self._hits += 1
                entry.hit_count += 1
                logger.debug(f"CACHE HIT fp={fp[:8]} (hits={self._hits})")
                return entry.response, entry.headers

        self._misses += 1
        logger.debug(f"CACHE MISS fp={fp[:8]} (misses={self._misses})")
        return None

    def set(self, body: dict, stream: bool, response: Any, headers: dict) -> None:
        """Cache a response."""
        # Skip caching for non-deterministic requests
        temperature = body.get("temperature")
        if temperature is not None and temperature > 0:
            logger.debug("Skipping cache — non-zero temperature")
            return

        fp = self._fingerprint(body, stream)
        self._cache[fp] = CacheEntry(response, headers, fp)
        logger.debug(f"CACHE SET fp={fp[:8]}")

    def invalidate(self, fingerprint_prefix: str) -> None:
        """Invalidate cache entries matching a prefix."""
        to_remove = [k for k in self._cache if k.startswith(fingerprint_prefix)]
        for k in to_remove:
            del self._cache[k]
        if to_remove:
            logger.info(f"CACHE INVALIDATE {len(to_remove)} entries")

    def _cleanup(self) -> None:
        """Remove expired entries."""
        expired = [k for k, v in self._cache.items() if v.is_expired(self._ttl)]
        for k in expired:
            del self._cache[k]

    @property
    def stats(self) -> dict:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "entries": len(self._cache),
            "ttl": self._ttl,
        }


# Global cache instance
_cache = ResponseCache()


def get_cache() -> ResponseCache:
    return _cache
