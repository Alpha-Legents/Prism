"""OpenAI-compatible provider plugin."""

import json
import logging
from typing import Any

from .base import ProviderPlugin

logger = logging.getLogger("prism.providers.openai_compat")


def translate_usage(usage: dict | None) -> dict:
    """Translate OpenAI-compatible usage to Anthropic usage.

    Cache-read tokens come from prompt_tokens_details.cached_tokens (OpenAI,
    OpenRouter, vLLM) or prompt_cache_hit_tokens (DeepSeek). Anthropic's
    input_tokens EXCLUDES cached reads while OpenAI's prompt_tokens includes
    them, so cached tokens are subtracted. Cache-creation is reported as 0:
    OpenAI-compatible providers do not bill cache writes separately, and
    double-reporting them would inflate client-side context-size accounting
    (clients sum input + cache_creation + cache_read + output).
    """
    usage = usage or {}
    prompt = usage.get("prompt_tokens") or 0
    completion = usage.get("completion_tokens") or 0
    details = usage.get("prompt_tokens_details") or {}
    cached = 0
    if isinstance(details, dict):
        cached = details.get("cached_tokens") or 0
    if not cached:
        cached = usage.get("prompt_cache_hit_tokens") or 0
    return {
        "input_tokens": max(prompt - cached, 0),
        "output_tokens": completion,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cached,
    }


class OpenAICompatPlugin(ProviderPlugin):
    """Plugin for OpenAI-compatible providers (Groq, Mistral, etc.)."""

    name = "openai-compat"

    def __init__(self):
        self._supports_tool_use = True
        self._supports_thinking = False

    def translate_request(self, body: dict, model_override: str | None = None) -> dict:
        """Translate Anthropic request to OpenAI-compatible format."""
        messages = self._translate_messages(body.get("messages", []), body.get("system"))
        tools = self._translate_tools(body.get("tools", []))

        req = {
            "model": model_override or body.get("model", "gpt-4o"),
            "messages": messages,
            "stream": body.get("stream", True),
        }

        # Optional fields - only add if present in original request
        if body.get("max_tokens") is not None:
            req["max_tokens"] = body["max_tokens"]
        if body.get("max_completion_tokens") is not None:
            req["max_completion_tokens"] = body["max_completion_tokens"]
        if body.get("temperature") is not None:
            req["temperature"] = body["temperature"]
        if body.get("top_p") is not None:
            req["top_p"] = body["top_p"]
        if body.get("top_k") is not None:
            req["top_k"] = body["top_k"]
        if body.get("stop_sequences"):
            # Anthropic stop_sequences -> OpenAI stop
            req["stop"] = body["stop_sequences"]
        if tools:
            req["tools"] = tools
        tc_raw = body.get("tool_choice")
        if tc_raw is not None:
            req["tool_choice"] = self._translate_tool_choice(tc_raw)
            # Anthropic disable_parallel_tool_use -> OpenAI parallel_tool_calls
            if isinstance(tc_raw, dict) and tc_raw.get("disable_parallel_tool_use"):
                req["parallel_tool_calls"] = False
        if body.get("response_format") is not None:
            req["response_format"] = body["response_format"]
        # Anthropic metadata.user_id -> OpenAI 'user' (the native equivalent).
        # Passing 'metadata' through verbatim breaks strict providers.
        meta = body.get("metadata")
        if isinstance(meta, dict) and meta.get("user_id"):
            req["user"] = str(meta["user_id"])

        return req

    def _translate_messages(self, messages: list, system: Any = None) -> list[dict]:
        """Convert Anthropic messages to OpenAI format."""
        out = []
        if system:
            # Anthropic system can be a plain string or a list of text blocks
            # (Claude Code sends TextBlockParam[] with cache_control).
            if isinstance(system, list):
                system_text = "\n\n".join(
                    b.get("text", "")
                    for b in system
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                system_text = str(system)
            if system_text:
                out.append({"role": "system", "content": system_text})

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            if isinstance(content, str):
                # Handle string content (assistant with pre-built tool_calls)
                if msg.get("tool_calls") and role == "assistant":
                    out.append({
                        "role": "assistant",
                        "content": content or "",
                        "tool_calls": msg["tool_calls"],
                    })
                else:
                    out.append({"role": role, "content": content or ""})
            elif isinstance(content, list):
                # Handle list of content blocks (Anthropic format)
                tool_results = []
                tool_uses = []
                text_parts = []
                image_parts = []

                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type", "")

                    if block_type == "tool_result":
                        tool_results.append(block)
                    elif block_type == "tool_use":
                        tool_uses.append(block)
                    elif block_type == "text":
                        text_parts.append(block.get("text", ""))
                    elif block_type in ("thinking", "redacted_thinking"):
                        # OpenAI-compatible providers have no thinking concept;
                        # drop silently (never echo thinking back as text).
                        continue
                    elif block_type == "image":
                        # Image URL handling
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            image_parts.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{source.get('media_type', 'image/jpeg')};base64,{source.get('data', '')}"
                                }
                            })
                        elif "url" in source:
                            image_parts.append({
                                "type": "image_url",
                                "image_url": {"url": source["url"]}
                            })

                # Emit tool results as separate messages
                if tool_results:
                    for tr in tool_results:
                        rc = tr.get("content", "")
                        # Handle list of content blocks in tool_result
                        if isinstance(rc, list):
                            text_content = []
                            for item in rc:
                                if not isinstance(item, dict):
                                    continue
                                if item.get("type") == "text":
                                    text_content.append(item.get("text", ""))
                                elif item.get("type") == "image":
                                    text_content.append("[image attached]")
                            rc = "\n".join(text_content)
                        out.append({
                            "role": "tool",
                            "tool_call_id": tr.get("tool_use_id", ""),
                            "content": rc or "",
                        })

                # Emit tool uses as assistant message (with any leading text)
                if tool_uses:
                    tool_calls = []
                    for tu in tool_uses:
                        input_data = tu.get("input", {})
                        if isinstance(input_data, dict):
                            args = json.dumps(input_data)
                        else:
                            args = str(input_data)
                        tool_calls.append({
                            "id": tu.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": tu.get("name", ""),
                                "arguments": args,
                            },
                        })

                    out.append({
                        "role": "assistant",
                        "content": "\n".join(text_parts) or "",
                        "tool_calls": tool_calls,
                    })

                # Emit remaining text/images. Text alongside tool_results
                # (e.g. interrupts, attachments) must NOT be dropped -- it is
                # emitted as its own message after the tool messages.
                if not tool_uses and (text_parts or image_parts or not tool_results):
                    if image_parts:
                        # Multimodal: text + images
                        multimodal_content = []
                        if text_parts:
                            multimodal_content.append({"type": "text", "text": "\n".join(text_parts)})
                        multimodal_content.extend(image_parts)
                        out.append({"role": role, "content": multimodal_content})
                    elif text_parts or not tool_results:
                        out.append({"role": role, "content": "\n".join(text_parts) or ""})
            else:
                out.append({"role": role, "content": str(content or "")})

        return out

    def _translate_tool_choice(self, tc: dict | str) -> str | dict:
        if isinstance(tc, dict):
            tc_type = tc.get("type", "auto")
            if tc_type == "any":
                return "required"
            elif tc_type == "tool":
                name = tc.get("name")
                if name:
                    return {"type": "function", "function": {"name": name}}
                return "auto"
            elif tc_type == "auto":
                return "auto"
            elif tc_type == "none":
                return "none"
        return tc

    def _translate_tools(self, tools: list) -> list[dict]:
        """Convert Anthropic tools to OpenAI format."""
        if not tools:
            return []
        result = []
        for t in tools:
            tool = {
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            fn = tool["function"]
            if t.get("strict"):
                fn["strict"] = True
            result.append(tool)
        return result

    def translate_response(self, raw: dict) -> dict:
        """Translate OpenAI response to Anthropic format."""
        choices = raw.get("choices", [{}])
        if not choices:
            return self._empty_response(raw)

        choice = choices[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "stop")
        usage = raw.get("usage", {})

        stop_reason = self._map_finish_reason(finish_reason)
        content_blocks = []

        # Reasoning models (DeepSeek reasoning_content, OpenRouter reasoning)
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if reasoning:
            content_blocks.append({
                "type": "thinking",
                "thinking": str(reasoning),
                "signature": "",
            })

        text = message.get("content", "")
        if text:
            content_blocks.append({"type": "text", "text": text})

        tool_calls = message.get("tool_calls", [])
        for tc in tool_calls:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments", "{}") or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {"_raw": fn.get("arguments", "")}
            content_blocks.append({
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{len(content_blocks)}"),
                "name": fn.get("name", ""),
                "input": args,
            })

        if tool_calls:
            stop_reason = "tool_use"

        return {
            "id": raw.get("id", "msg_prism"),
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "model": raw.get("model", "unknown"),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": translate_usage(usage),
        }

    def translate_stream_chunk(self, chunk: dict) -> list[dict]:
        """Not used directly streaming is handled in translate/stream.py."""
        return []

    def _map_finish_reason(self, finish_reason: str | None) -> str:
        mapping = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
            "content_filter": "refusal",
            "function_call": "tool_use",
        }
        return mapping.get(finish_reason, finish_reason or "end_turn")

    def _empty_response(self, raw: dict) -> dict:
        usage = raw.get("usage", {})
        return {
            "id": raw.get("id", "msg_prism_empty"),
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": ""}],
            "model": raw.get("model", "unknown"),
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": translate_usage(usage),
        }

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_tool_use(self) -> bool:
        return self._supports_tool_use

    @property
    def supports_thinking(self) -> bool:
        return self._supports_thinking
