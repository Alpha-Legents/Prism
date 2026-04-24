"""
Request translator.
Converts client-shaped requests → provider-shaped requests.
Messages, tools, tool_results are NEVER touched — only envelope.
"""

import logging
from ..slots import extract, REQUEST_SLOTS, detect_format

logger = logging.getLogger("prism.translate.request")


def translate_request(
    body: dict,
    client_format: str,
    provider_format: str,
    model_override: str | None = None,
) -> dict:
    """
    Translate a client request body to provider format.

    Args:
        body:             Raw request from client tool
        client_format:    Detected format of the client ('anthropic'|'openai-compat')
        provider_format:  Target provider format
        model_override:   If set, replace whatever model the client sent
    """
    if client_format == provider_format and not model_override:
        # Same format — just override model if needed and pass through
        if model_override:
            return {**body, "model": model_override}
        return body

    if provider_format == "openai-compat":
        return _to_openai(body, client_format, model_override)
    elif provider_format == "anthropic":
        return _to_anthropic(body, client_format, model_override)
    else:
        logger.warning(f"Unknown provider format {provider_format} — passing through")
        return body


def _flatten_system(system) -> str | None:
    """Flatten system prompt — could be string or Anthropic content block array."""
    if system is None:
        return None
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts).strip() or None
    return str(system)


def _to_openai(body: dict, src: str, model_override: str | None) -> dict:
    """Translate any client request → OpenAI Chat Completions request."""

    messages    = extract(body, "messages",    REQUEST_SLOTS) or []
    max_tokens  = extract(body, "max_tokens",  REQUEST_SLOTS)
    tools       = extract(body, "tools",       REQUEST_SLOTS)
    stream      = extract(body, "stream",      REQUEST_SLOTS)
    temperature = extract(body, "temperature", REQUEST_SLOTS)
    top_p       = extract(body, "top_p",       REQUEST_SLOTS)
    model       = model_override or extract(body, "model", REQUEST_SLOTS) or "gpt-4o"
    system      = _flatten_system(body.get("system"))

    out: dict = {
        "model":    model,
        "messages": _messages_to_openai(messages, system),
    }

    if max_tokens  is not None: out["max_tokens"]   = max_tokens
    if tools:                   out["tools"]         = _tools_to_openai(tools, src)
    if stream      is not None: out["stream"]        = stream
    if temperature is not None: out["temperature"]   = temperature
    if top_p       is not None: out["top_p"]         = top_p

    return out


def _to_anthropic(body: dict, src: str, model_override: str | None) -> dict:
    """Translate any client request → Anthropic Messages API request."""

    messages    = extract(body, "messages",    REQUEST_SLOTS) or []
    max_tokens  = extract(body, "max_tokens",  REQUEST_SLOTS) or 1024
    tools       = extract(body, "tools",       REQUEST_SLOTS)
    stream      = extract(body, "stream",      REQUEST_SLOTS)
    temperature = extract(body, "temperature", REQUEST_SLOTS)
    model       = model_override or extract(body, "model", REQUEST_SLOTS) or "claude-sonnet-4-20250514"

    # Extract system prompt — Anthropic has it top-level
    system = body.get("system") or _extract_system_from_messages(messages)
    clean_messages = _strip_system_from_messages(messages) if not body.get("system") else messages

    out: dict = {
        "model":      model,
        "max_tokens": max_tokens,
        "messages":   _messages_to_anthropic(clean_messages),
    }

    if system:                  out["system"]      = system
    if tools:                   out["tools"]       = _tools_to_anthropic(tools, src)
    if stream is not None:      out["stream"]      = stream
    if temperature is not None: out["temperature"] = temperature

    return out


# ── Message format converters ─────────────────────────────────────────────────

def _flatten_content_blocks(blocks: list) -> str:
    """Flatten Anthropic content block array → plain string for OpenAI."""
    parts = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            parts.append(b.get("text", ""))
        elif b.get("type") == "tool_result":
            rc = b.get("content", "")
            if isinstance(rc, list):
                rc = " ".join(x.get("text", "") for x in rc if isinstance(x, dict))
            parts.append(f"[tool_result: {rc}]")
    return "\n".join(parts)


def _messages_to_openai(messages: list, system: str | None) -> list:
    """Convert messages (any format) → OpenAI messages array."""
    import json as _json
    out = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        role    = m.get("role", "user")
        content = m.get("content")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Anthropic content blocks → OpenAI
            tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
            tool_uses    = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
            text_parts   = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]

            if tool_results:
                # Each tool_result → separate OpenAI "tool" message
                for tr in tool_results:
                    rc = tr.get("content", "")
                    if isinstance(rc, list):
                        rc = "\n".join(
                            x.get("text", "") for x in rc
                            if isinstance(x, dict) and x.get("type") == "text"
                        )
                    out.append({
                        "role":         "tool",
                        "tool_call_id": tr.get("tool_use_id", ""),
                        "content":      rc or "",
                    })
            elif tool_uses:
                # Assistant message with tool calls
                oai_msg: dict = {"role": "assistant", "content": " ".join(text_parts) or None}
                oai_msg["tool_calls"] = [
                    {
                        "id":   tu.get("id", ""),
                        "type": "function",
                        "function": {
                            "name":      tu.get("name", ""),
                            "arguments": _json.dumps(tu.get("input", {})),
                        },
                    }
                    for tu in tool_uses
                ]
                out.append(oai_msg)
            else:
                out.append({"role": role, "content": " ".join(text_parts)})
        else:
            out.append({"role": role, "content": str(content or "")})

    return out


def _messages_to_anthropic(messages: list) -> list:
    """Convert messages (any format) → Anthropic messages array."""
    out = []
    i = 0
    while i < len(messages):
        m    = messages[i]
        role = m.get("role", "user")
        content = m.get("content")

        if role == "system":
            i += 1
            continue  # system handled top-level

        if role == "tool":
            # OpenAI tool result → Anthropic tool_result block
            out.append({
                "role": "user",
                "content": [{
                    "type":        "tool_result",
                    "tool_use_id": m.get("tool_call_id", ""),
                    "content":     content or "",
                }],
            })
            i += 1
            continue

        if isinstance(content, str):
            # Check if this is an assistant message with tool_calls
            tool_calls = m.get("tool_calls")
            if tool_calls and role == "assistant":
                blocks = []
                if content:
                    blocks.append({"type": "text", "text": content})
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    import json
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except Exception:
                        args = {}
                    blocks.append({
                        "type":  "tool_use",
                        "id":    tc.get("id", ""),
                        "name":  fn.get("name", ""),
                        "input": args,
                    })
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Already Anthropic content blocks — pass through
            out.append({"role": role, "content": content})
        else:
            out.append({"role": role, "content": str(content or "")})

        i += 1

    return out


def _extract_system_from_messages(messages: list) -> str | None:
    for m in messages:
        if m.get("role") == "system":
            return m.get("content", "")
    return None


def _strip_system_from_messages(messages: list) -> list:
    return [m for m in messages if m.get("role") != "system"]


# ── Tool format converters ────────────────────────────────────────────────────

def _tools_to_openai(tools: list, src: str) -> list:
    if src == "anthropic":
        # Anthropic tools → OpenAI function tools
        return [
            {
                "type": "function",
                "function": {
                    "name":        t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters":  t.get("input_schema", {}),
                },
            }
            for t in tools
        ]
    return tools  # Already OpenAI format


def _tools_to_anthropic(tools: list, src: str) -> list:
    if src == "openai-compat":
        # OpenAI function tools → Anthropic tools
        return [
            {
                "name":         t.get("function", {}).get("name", ""),
                "description":  t.get("function", {}).get("description", ""),
                "input_schema": t.get("function", {}).get("parameters", {}),
            }
            for t in tools
            if t.get("type") == "function"
        ]
    return tools  # Already Anthropic format