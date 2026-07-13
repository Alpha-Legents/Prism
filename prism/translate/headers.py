"""Header translation — maps provider headers → client-expected headers."""

import logging

logger = logging.getLogger("prism.translate.headers")

# Headers to strip from provider responses (noisy/hop-by-hop)
_STRIP_HEADERS = {
    "server", "via", "x-powered-by", "cf-cache-status",
    "alt-svc", "nel", "report-to", "x-envoy-upstream-service-time",
    "content-length", "content-encoding", "transfer-encoding",
}

# Critical beta headers that MUST be preserved regardless of provider
# These are essential for Claude Code functionality
PRESERVED_BETA_HEADERS = {
    # Prompt caching
    "prompt-caching-2024-07-01",
    "prompt-caching-2025-07-01",
    # Model-specific features
    "max-tokens-3-5-sonnet-2024-07-15",
    "computer-use-2024-10-22",
    "computer-use-2025-01-10",
    "output-128k-2025-02-19",
    # Thinking
    "thinking-2024-11-20",
    "thinking-2025-01-21",
    # Context
    "context-1m-2025-07-01",
    "context-1m-2025-08-07",
    "context-management-2025-06-01",
    "context-management-2025-06-27",
    # Structured outputs
    "structured-outputs-2025-06-01",
    "structured-outputs-2025-12-15",
    # Token efficient tools
    "token-efficient-tools-2025-06-01",
    "token-efficient-tools-2026-03-28",
    # Other critical
    "fast-mode-2025-06-01",
    "fast-mode-2026-02-01",
    "effort-2025-06-01",
    "effort-2025-11-24",
    "task-budgets-2026-03-13",
    "redact-thinking-2025-06-01",
    "redact-thinking-2026-02-12",
    "interleaved-thinking-2025-05-14",
    "web-search-2025-03-05",
    "advanced-tool-use-2025-11-20",
    "tool-search-tool-2025-10-19",
    "summarize-connector-text-2026-03-13",
    "afk-mode-2026-01-31",
    "advisor-tool-2026-03-01",
    "environments-2025-11-01",
    "ccr-byoc-2025-07-29",
    # OAuth
    "oauth-2025-04-20",
    # Cache editing
    "cache-editing-2025-08-07",
    "prompt-caching-scope-2026-01-05",
    "cached-microcompact-2025-06-01",
}

# Provider-specific supported betas (providers that DO support these)
PROVIDER_SUPPORTED_BETAS = {
    "anthropic": PRESERVED_BETA_HEADERS,
    "openai-compat": set(),  # Default: strip all, but we'll be smart about it
    "gemini": set(),
}


def translate_headers(
    incoming: dict,
    client_format: str,
    provider_format: str = "openai-compat",
) -> dict:
    """
    Translate provider response headers to client-expected format.
    Preserves critical beta headers needed by Claude Code.
    """
    normalized = {str(k).lower(): v for k, v in incoming.items()}

    # Start with incoming headers minus stripped ones
    out = {k: v for k, v in normalized.items() if k not in _STRIP_HEADERS}

    # Handle anthropic-beta header carefully
    if "anthropic-beta" in normalized:
        betas = str(normalized["anthropic-beta"]).split(",")
        beta_set = frozenset(b.strip() for b in betas)

        # Always preserve critical beta headers
        # Only filter out betas that are known to be unsupported
        allowed = PROVIDER_SUPPORTED_BETAS.get(provider_format, set())

        filtered = []
        for b in beta_set:
            b_clean = b.strip()
            if not b_clean:
                continue
            if b_clean in PRESERVED_BETA_HEADERS:
                # Always preserve critical betas
                filtered.append(b_clean)
            elif allowed and b_clean in allowed:
                filtered.append(b_clean)
            elif not allowed:
                # If provider doesn't specify, keep unknown betas (safer)
                filtered.append(b_clean)

        if filtered:
            out["anthropic-beta"] = ",".join(filtered)
            logger.debug(f"Preserved beta headers: {filtered}")
        else:
            # Remove entirely if nothing left
            out.pop("anthropic-beta", None)

    # Inject format-specific headers
    if client_format == "anthropic":
        out["anthropic-version"] = "2023-06-01"
    
    out["content-type"] = "application/json"
    out["content-encoding"] = "identity"
    out["cache-control"] = "no-cache"
    out["x-accel-buffering"] = "no"

    return out
