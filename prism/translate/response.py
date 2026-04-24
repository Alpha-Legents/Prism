"""
Response translator.
Converts provider-shaped responses → client-expected format.
Tool call internals are NEVER touched.
"""

import json
import logging
from ..slots import extract, RESPONSE_SLOTS, TO_ANTHROPIC, TO_OPENAI, detect_format

logger = logging.getLogger("prism.translate.response")


def translate_response(raw: dict, client_format: str) -> dict:
    """
    Translate provider response → client format.

    Args:
        raw:           Raw response from provider
        client_format: What the client expects ('anthropic'|'openai-compat')
    """
    provider_format = detect_format(raw)

    if provider_format == client_format:
        return raw  # Already correct shape

    if client_format == "anthropic":
        return _to_anthropic(raw, provider_format)
    elif client_format == "openai-compat":
        return _to_openai(raw, provider_format)
    else:
        return raw


def _to_anthropic(raw: dict, src: str) -> dict:
    stop_raw      = extract(raw, "stop_reason") or "stop"
    stop_reason   = TO_ANTHROPIC.get(stop_raw, "end_turn")
    input_tokens  = extract(raw, "input_tokens")  or 0
    output_tokens = extract(raw, "output_tokens") or 0
    model         = extract(raw, "model")         or "unknown"
    resp_id       = extract(raw, "response_id")   or "msg_prism"

    content_blocks: list[dict] = []

    if src == "openai-compat":
        text       = extract(raw, "response_text")
        tool_calls = extract(raw, "tool_calls_openai")

        if text:
            content_blocks.append({"type": "text", "text": text})

        if tool_calls:
            stop_reason = "tool_use"
            for tc in tool_calls:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except Exception:
                    args = {"_raw": fn.get("arguments", "")}
                content_blocks.append({
                    "type":  "tool_use",
                    "id":    tc.get("id", f"toolu_{len(content_blocks)}"),
                    "name":  fn.get("name", ""),
                    "input": args,
                })

    elif src == "anthropic":
        content_blocks = raw.get("content", [])

    return {
        "id":            resp_id,
        "type":          "message",
        "role":          "assistant",
        "content":       content_blocks,
        "model":         model,
        "stop_reason":   stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
        },
    }


def _to_openai(raw: dict, src: str) -> dict:
    stop_raw      = extract(raw, "stop_reason") or "end_turn"
    finish_reason = TO_OPENAI.get(stop_raw, stop_raw)
    input_tokens  = extract(raw, "input_tokens")  or 0
    output_tokens = extract(raw, "output_tokens") or 0
    model         = extract(raw, "model")         or "unknown"
    resp_id       = extract(raw, "response_id")   or "chatcmpl_prism"

    message: dict = {"role": "assistant"}

    if src == "anthropic":
        blocks     = raw.get("content", [])
        text_parts = [b["text"] for b in blocks if b.get("type") == "text"]
        tool_uses  = [b for b in blocks if b.get("type") == "tool_use"]

        message["content"] = text_parts[0] if text_parts else None
        if tool_uses:
            finish_reason     = "tool_calls"
            message["tool_calls"] = [
                {
                    "id":   tu.get("id", ""),
                    "type": "function",
                    "function": {
                        "name":      tu.get("name", ""),
                        "arguments": json.dumps(tu.get("input", {})),
                    },
                }
                for tu in tool_uses
            ]
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
