"""
Client probe.
Learns what format a client tool (Claude Code, Aider, Cursor, etc.)
sends requests in — by intercepting the first real request.

We can't fire an outbound probe at a client tool since it's the one
calling US. So we use a passive learning approach:
- Start listening
- First request that arrives gets fingerprinted
- Schema locked in, used for all subsequent translations
"""

import logging
from dataclasses import dataclass, field
from ..slots import detect_format

logger = logging.getLogger("prism.probe.client")


@dataclass
class ClientSchema:
    format: str = "unknown"           # 'anthropic' | 'openai-compat'
    sample_request: dict = field(default_factory=dict)
    learned: bool = False

    # Known client tool signatures for display
    tool_hint: str = "unknown"

    def __repr__(self):
        return f"<Client format={self.format} tool={self.tool_hint} learned={self.learned}>"


# Known client fingerprints
# If a request contains these keys/values, we know what tool it is
CLIENT_FINGERPRINTS: list[tuple[str, str]] = [
    ("user-agent", "claude-code"),
    ("user-agent", "aider"),
    ("user-agent", "cursor"),
    ("user-agent", "continue"),
    ("user-agent", "opencode"),
    ("user-agent", "cline"),
    ("x-stainless-package-version", ""),   # Claude Code SDK marker
]


def learn_from_request(body: dict, headers: dict) -> ClientSchema:
    """
    Fingerprint an incoming request to learn client format.
    Called on first request, schema locked in after.
    """
    schema = ClientSchema()

    # Detect format from body shape
    schema.format        = detect_format(body)
    schema.sample_request = body
    schema.learned        = True

    # Try to identify the tool from headers
    ua = headers.get("user-agent", "").lower()
    schema.tool_hint = _identify_tool(ua, headers)

    logger.info(f"Client learned: {schema}")
    return schema


def _identify_tool(ua: str, headers: dict) -> str:
    if "claude" in ua and "code" in ua:
        return "claude-code"
    if "aider" in ua:
        return "aider"
    if "cursor" in ua:
        return "cursor"
    if "continue" in ua:
        return "continue"
    if "opencode" in ua:
        return "opencode"
    if "cline" in ua:
        return "cline"
    # Claude Code SDK marker
    if "x-stainless-package-version" in {k.lower() for k in headers}:
        return "claude-code"
    # Anthropic format without UA → probably Claude Code
    if "anthropic-version" in {k.lower() for k in headers}:
        return "claude-code (assumed)"
    return "unknown"
