"""
Response translator (non-streaming).
Converts provider-shaped responses → client-expected format.
Tool call internals are NEVER touched.

The openai-compat → anthropic direction delegates to the provider plugin so
there is exactly one implementation of that mapping (usage/cache token
semantics, thinking blocks, refusal stop reasons).
"""

import json
import logging

from ..providers.openai_compat import OpenAICompatPlugin, translate_usage
from ..slots import extract, TO_OPENAI, detect_format
from .errors import translate_error  # re-exported for backwards compatibility

logger = logging.getLogger("prism.translate.response")

_plugin = OpenAICompatPlugin()


def translate_response(raw: dict, client_format: str, request_uuids: list = None) -> dict:
    """
    Translate provider response → client format.

    Args:
        raw:           Raw response from provider
        client_format: What the client expects ('anthropic'|'openai-compat')
        request_uuids: Unused, kept for backwards compatibility
    """
    provider_format = detect_format(raw)

    if provider_format == client_format:
        return raw  # Already correct shape

    if client_format == "anthropic":
        if provider_format == "openai-compat":
            return _plugin.translate_response(raw)
        return raw
    if client_format == "openai-compat":
        return _to_openai(raw, provider_format)
    return raw


def _to_openai(raw: dict, src: str) -> dict:
    stop_raw      = extract(raw, "stop_reason") or "end_turn"
    finish_reason = TO_OPENAI.get(stop_raw, "stop")
    input_tokens  = extract(raw, "input_tokens")  or 0
    output_tokens = extract(raw, "output_tokens") or 0
    model         = extract(raw, "model")         or "unknown"
    resp_id       = extract(raw, "response_id")   or "chatcmpl_prism"

    message: dict = {"role": "assistant"}

    if src == "anthropic":
        blocks         = raw.get("content", [])
        text_parts     = [b["text"] for b in blocks if b.get("type") == "text"]
        thinking_parts = [b.get("thinking", "") for b in blocks if b.get("type") == "thinking"]
        tool_uses      = [b for b in blocks if b.get("type") == "tool_use"]

        message["content"] = "\n".join(text_parts) if text_parts else ""
        if thinking_parts:
            # Expose thinking via the de-facto OpenAI-compat reasoning field
            message["reasoning_content"] = "\n".join(thinking_parts)
        if tool_uses:
            # In OpenAI, tool use triggers the tool_calls finish reason
            finish_reason = "tool_calls"
            message["tool_calls"] = []
            for tu in tool_uses:
                try:
                    args = json.dumps(tu.get("input", {}))
                except Exception:
                    args = "{}"
                message["tool_calls"].append({
                    "id":   tu.get("id", ""),
                    "type": "function",
                    "function": {
                        "name":      tu.get("name", ""),
                        "arguments": args,
                    }
                })
    elif src == "openai-compat":
        choices = raw.get("choices", [{}])
        message = choices[0].get("message", {"role": "assistant", "content": ""})

    return {
        "id":      resp_id,
        "object":  "chat.completion",
        "model":   model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens":     input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens":      input_tokens + output_tokens,
        },
    }


# ── Context-window detection (kept for backwards compatibility) ────────────
CONTEXT_PATTERNS = [
    "maximum context length",
    "context window",
    "token limit",
    "context too long",
    "context_length_exceeded",
]


def is_context_window_exceeded(raw: dict) -> bool:
    """Detect context window exceeded errors from any provider."""
    error = raw.get("error", raw)
    if not isinstance(error, dict):
        return False
    error_type = error.get("type", "") or ""
    error_msg = (error.get("message", "") or "").lower()

    if "context_length_exceeded" in error_type.lower():
        return True
    return any(p in error_msg for p in CONTEXT_PATTERNS)
