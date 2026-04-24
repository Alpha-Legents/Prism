"""
Semantic slot definitions.
Maps canonical concepts → jmespath paths across every known provider.
This is the Rosetta Stone of prism.
"""

from typing import Any
import jmespath

# ── Response slots (what providers return) ───────────────────────────────────
RESPONSE_SLOTS: dict[str, list[str]] = {
    "response_text": [
        "choices[0].message.content",
        "content[0].text",
        "output[0].text",
        "candidates[0].content.parts[0].text",
    ],
    "stop_reason": [
        "choices[0].finish_reason",
        "stop_reason",
        "candidates[0].finishReason",
    ],
    "input_tokens": [
        "usage.prompt_tokens",
        "usage.input_tokens",
        "usageMetadata.promptTokenCount",
    ],
    "output_tokens": [
        "usage.completion_tokens",
        "usage.output_tokens",
        "usageMetadata.candidatesTokenCount",
    ],
    "model":       ["model", "modelVersion"],
    "response_id": ["id"],
    "tool_calls_openai":    ["choices[0].message.tool_calls"],
    "tool_calls_anthropic": ["content[?type=='tool_use']"],
    "content_blocks":       ["content"],
    "message_role":         ["choices[0].message.role", "role"],
}

# ── Request slots (what client tools send) ───────────────────────────────────
REQUEST_SLOTS: dict[str, list[str]] = {
    "messages":   ["messages"],
    "model":      ["model"],
    "max_tokens": ["max_tokens"],
    "tools":      ["tools"],
    "system":     ["system"],
    "stream":     ["stream"],
    "temperature":["temperature"],
    "top_p":      ["top_p"],
}

# ── Finish reason normalization maps ─────────────────────────────────────────
TO_ANTHROPIC: dict[str, str] = {
    "stop":           "end_turn",
    "tool_calls":     "tool_use",
    "length":         "max_tokens",
    "content_filter": "stop_sequence",
    "end_turn":       "end_turn",
    "tool_use":       "tool_use",
    "max_tokens":     "max_tokens",
}

TO_OPENAI: dict[str, str] = {v: k for k, v in TO_ANTHROPIC.items()}


def extract(data: dict, slot: str, pool: dict = RESPONSE_SLOTS) -> Any:
    """Try each known path for a slot until one hits."""
    for path in pool.get(slot, []):
        try:
            val = jmespath.search(path, data)
            if val is not None:
                return val
        except Exception:
            continue
    return None


def detect_format(data: dict) -> str:
    """
    Identify the wire format of a message (request OR response).
    Returns: 'anthropic' | 'openai-compat' | 'gemini' | 'unknown'
    """
    # Response detection
    if "choices" in data:
        return "openai-compat"
    if "stop_reason" in data and "content" in data:
        return "anthropic"
    if "candidates" in data:
        return "gemini"
    # Request detection
    if "messages" in data:
        if "anthropic-version" in str(data) or "system" in data:
            return "anthropic"
        return "openai-compat"   # most clients default to this
    return "unknown"
