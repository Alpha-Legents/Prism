"""
Streaming translator.
Converts OpenAI SSE stream → Anthropic SSE stream on the fly.

Strategy:
  - Text chunks      → stream as content_block_delta immediately
  - Thinking chunks  → stream as thinking_block_delta immediately
  - Tool calls       → buffer per tool index, emit complete at end
  - finish_reason    → emit message_delta + message_stop

Anthropic SSE event sequence expected by Claude Code:
  message_start
  content_block_start (index 0)
  [content_block_delta ...]
  content_block_stop
  [more blocks if tools...]
  message_delta (with stop_reason)
  message_stop
"""

import json
import logging
from typing import AsyncIterator
from ..probe.capabilities import get_capabilities, learn_from_chunk, extract_thinking_text

logger = logging.getLogger("prism.translate.stream")


def _sse(event: str, data: dict) -> str:
    """Format a single SSE event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def translate_stream(
    chunks: AsyncIterator[bytes],
    client_format: str,
    model: str,
    msg_id: str = "msg_prism",
) -> AsyncIterator[str]:
    """
    Translate an OpenAI SSE stream → Anthropic SSE stream.
    Yields SSE-formatted strings ready to send to client.
    """
    if client_format != "anthropic":
        # For non-Anthropic clients just pass through raw chunks
        async for chunk in chunks:
            yield chunk.decode("utf-8", errors="replace")
        return

    caps               = get_capabilities(model)
    text_buf           = ""
    thinking_buf       = ""
    tool_bufs: dict[int, dict] = {}  # index → {id, name, args_buf}
    has_text           = False
    has_thinking       = False
    content_block_idx  = 0
    input_tokens       = 0
    output_tokens      = 0
    finish_reason      = "end_turn"
    started            = False

    async for raw_chunk in chunks:
        line = raw_chunk.decode("utf-8", errors="replace").strip()

        if not line or line == "data: [DONE]":
            continue

        if not line.startswith("data:"):
            continue

        try:
            chunk = json.loads(line[5:].strip())
        except Exception:
            continue

        # Learn capabilities from this chunk
        learn_from_chunk(model, chunk)

        choice      = (chunk.get("choices") or [{}])[0]
        delta       = choice.get("delta") or {}
        fin_reason  = choice.get("finish_reason")

        # Track token usage
        usage = chunk.get("usage") or {}
        if usage.get("prompt_tokens"):    input_tokens  = usage["prompt_tokens"]
        if usage.get("completion_tokens"): output_tokens = usage["completion_tokens"]

        # ── Emit message_start on first real chunk ────────────────────────────
        if not started:
            started = True
            yield _sse("message_start", {
                "type": "message_start",
                "message": {
                    "id":    msg_id,
                    "type":  "message",
                    "role":  "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason":   None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                },
            })

        # ── Handle thinking delta ─────────────────────────────────────────────
        # DeepSeek/QwQ style: reasoning_content in delta
        reasoning = delta.get("reasoning_content")
        if reasoning:
            if not has_thinking:
                has_thinking = True
                yield _sse("content_block_start", {
                    "type":  "content_block_start",
                    "index": content_block_idx,
                    "content_block": {"type": "thinking", "thinking": ""},
                })
            thinking_buf += reasoning
            yield _sse("content_block_delta", {
                "type":  "content_block_delta",
                "index": content_block_idx,
                "delta": {"type": "thinking_delta", "thinking": reasoning},
            })
            continue

        # Content array style (Mistral vibe models)
        delta_content = delta.get("content")
        if isinstance(delta_content, list):
            for block in delta_content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")

                if btype in ("thinking", "reasoning"):
                    think_text = extract_thinking_text(block, caps) or ""
                    if not has_thinking:
                        has_thinking = True
                        yield _sse("content_block_start", {
                            "type":  "content_block_start",
                            "index": content_block_idx,
                            "content_block": {"type": "thinking", "thinking": ""},
                        })
                    thinking_buf += think_text
                    yield _sse("content_block_delta", {
                        "type":  "content_block_delta",
                        "index": content_block_idx,
                        "delta": {"type": "thinking_delta", "thinking": think_text},
                    })

                elif btype == "text":
                    txt = block.get("text", "")
                    if txt:
                        if not has_text:
                            # Close thinking block if open
                            if has_thinking:
                                yield _sse("content_block_stop", {"type": "content_block_stop", "index": content_block_idx})
                                content_block_idx += 1
                            has_text = True
                            yield _sse("content_block_start", {
                                "type":  "content_block_start",
                                "index": content_block_idx,
                                "content_block": {"type": "text", "text": ""},
                            })
                        text_buf += txt
                        yield _sse("content_block_delta", {
                            "type":  "content_block_delta",
                            "index": content_block_idx,
                            "delta": {"type": "text_delta", "text": txt},
                        })
            continue

        # ── Plain text delta ──────────────────────────────────────────────────
        if isinstance(delta_content, str) and delta_content:
            if not has_text:
                if has_thinking:
                    yield _sse("content_block_stop", {"type": "content_block_stop", "index": content_block_idx})
                    content_block_idx += 1
                has_text = True
                yield _sse("content_block_start", {
                    "type":  "content_block_start",
                    "index": content_block_idx,
                    "content_block": {"type": "text", "text": ""},
                })
            text_buf += delta_content
            yield _sse("content_block_delta", {
                "type":  "content_block_delta",
                "index": content_block_idx,
                "delta": {"type": "text_delta", "text": delta_content},
            })

        # ── Tool call deltas — buffer, don't stream ───────────────────────────
        tool_calls = delta.get("tool_calls") or []
        for tc in tool_calls:
            idx  = tc.get("index", 0)
            if idx not in tool_bufs:
                tool_bufs[idx] = {
                    "id":       tc.get("id", f"toolu_{idx}"),
                    "name":     (tc.get("function") or {}).get("name", ""),
                    "args_buf": "",
                }
            fn = tc.get("function") or {}
            if fn.get("name"):
                tool_bufs[idx]["name"] = fn["name"]
            if fn.get("arguments"):
                tool_bufs[idx]["args_buf"] += fn["arguments"]

        # ── Handle finish ─────────────────────────────────────────────────────
        if fin_reason:
            finish_reason = {
                "tool_calls": "tool_use",
                "stop":       "end_turn",
                "length":     "max_tokens",
            }.get(fin_reason, "end_turn")

    # ── Emit buffered tool calls ──────────────────────────────────────────────
    if tool_bufs:
        finish_reason = "tool_use"

        # Close open text/thinking block first
        if has_text or has_thinking:
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": content_block_idx})
            content_block_idx += 1

        for idx in sorted(tool_bufs.keys()):
            tc = tool_bufs[idx]
            try:    args = json.loads(tc["args_buf"] or "{}")
            except: args = {"_raw": tc["args_buf"]}

            yield _sse("content_block_start", {
                "type":  "content_block_start",
                "index": content_block_idx,
                "content_block": {
                    "type":  "tool_use",
                    "id":    tc["id"],
                    "name":  tc["name"],
                    "input": {},
                },
            })
            yield _sse("content_block_delta", {
                "type":  "content_block_delta",
                "index": content_block_idx,
                "delta": {"type": "input_json_delta", "partial_json": tc["args_buf"]},
            })
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": content_block_idx})
            content_block_idx += 1

    elif has_text or has_thinking:
        # Close the last open block
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": content_block_idx})

    # ── message_delta + message_stop ─────────────────────────────────────────
    if not started:
        # Empty response — still need to emit message_start
        yield _sse("message_start", {
            "type": "message_start",
            "message": {
                "id": msg_id, "type": "message", "role": "assistant",
                "model": model, "content": [], "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        })

    yield _sse("message_delta", {
        "type":  "message_delta",
        "delta": {"stop_reason": finish_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })
    yield _sse("message_stop", {"type": "message_stop"})

    logger.info(f"STREAM done: thinking={has_thinking} text={has_text} tools={len(tool_bufs)} stop={finish_reason}")