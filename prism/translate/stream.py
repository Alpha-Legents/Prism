"""
Robust SSE translator: OpenAI-compatible streaming chunks -> Anthropic SSE.

Event contract expected by Anthropic-SDK clients (e.g. Claude Code):

  1. message_start
  2. ping
  3. content_block_start / content_block_delta / content_block_stop
     (per block, strictly sequential, indexes announced before use)
  4. message_delta   (final stop_reason + usage)
  5. message_stop

Invariants this translator guarantees (violating any of these makes strict
clients throw or discard the response):

- Every content_block_delta references an index previously announced by a
  content_block_start (clients raise otherwise).
- Delta types always match their block type:
  text_delta->text, input_json_delta->tool_use, thinking_delta->thinking.
- Every started block is closed with a content_block_stop. Clients only
  materialize assistant messages on content_block_stop.
- message_delta always carries a stop_reason. A stream that ends without one
  is treated as a failed stream by clients and triggers fallback retries.
- Message ids are unique per response. Clients group split assistant records
  by message id, so reusing ids across responses corrupts token accounting.

Supported provider delta shapes (OpenAI-compatible superset):
- delta.content                              -> text block
- delta.reasoning_content / delta.reasoning -> thinking block
  (DeepSeek, OpenRouter, Qwen, ...)
- delta.tool_calls[]                         -> tool_use blocks with
  input_json_delta (buffered until the tool name is known)
- chunk.usage                                -> real token usage
  (stream_options.include_usage, Mistral/DeepSeek final-chunk usage, ...)
- chunk.error                                -> Anthropic error event
"""

import json
import logging
from typing import AsyncIterator, Optional
from uuid import uuid4

from ..providers.openai_compat import translate_usage
from .errors import translate_error

logger = logging.getLogger("prism.translate.stream")

# OpenAI finish_reason -> Anthropic stop_reason
FINISH_REASON_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


def _sse(event_type: str, data: dict) -> str:
    """Format a single SSE event with proper double-newline termination."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


class _AnthropicStreamBuilder:
    """Sequential Anthropic content-block state machine.

    Only one block is open at any time. Text, thinking, and tool_use deltas
    are routed to the open block; a type switch closes the current block and
    opens a new one at the next index. Tool calls whose name has not yet
    arrived are buffered so content_block_start always carries the real name.
    """

    def __init__(self) -> None:
        self.next_index = 0
        self.open_kind: Optional[str] = None       # 'text' | 'thinking' | 'tool'
        self.open_index: Optional[int] = None
        self.open_tc_index: Optional[int] = None   # provider index of open tool
        self.started_tools: set[int] = set()       # provider tool indexes started
        self.pending_tools: dict[int, dict] = {}   # awaiting a name before start
        self.saw_tool_block = False
        self.output_chars = 0

    # -- block lifecycle -----------------------------------------------------

    def close_open(self, events: list[str]) -> None:
        if self.open_index is not None:
            # For thinking blocks, emit signature_delta before content_block_stop.
            # Claude Code requires a signature on every thinking block — without
            # it the SDK throws "Content block is not a thinking block" or
            # silently discards the thinking, causing context loss.
            if self.open_kind == "thinking":
                events.append(_sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": self.open_index,
                    "delta": {"type": "signature_delta", "signature": ""},
                }))
            events.append(_sse("content_block_stop", {
                "type": "content_block_stop",
                "index": self.open_index,
            }))
        self.open_kind = None
        self.open_index = None
        self.open_tc_index = None

    def _open(
        self,
        events: list[str],
        kind: str,
        content_block: dict,
        tc_index: Optional[int] = None,
    ) -> None:
        self.close_open(events)
        index = self.next_index
        self.next_index += 1
        events.append(_sse("content_block_start", {
            "type": "content_block_start",
            "index": index,
            "content_block": content_block,
        }))
        self.open_kind = kind
        self.open_index = index
        self.open_tc_index = tc_index

    # -- deltas ---------------------------------------------------------------

    def text(self, events: list[str], text: str) -> None:
        if self.open_kind != "text":
            self._open(events, "text", {"type": "text", "text": ""})
        events.append(_sse("content_block_delta", {
            "type": "content_block_delta",
            "index": self.open_index,
            "delta": {"type": "text_delta", "text": text},
        }))
        self.output_chars += len(text)

    def thinking(self, events: list[str], text: str) -> None:
        if self.open_kind != "thinking":
            self._open(events, "thinking", {
                "type": "thinking",
                "thinking": "",
                "signature": "",
            })
        events.append(_sse("content_block_delta", {
            "type": "content_block_delta",
            "index": self.open_index,
            "delta": {"type": "thinking_delta", "thinking": text},
        }))
        self.output_chars += len(text)

    def tool_call(self, events: list[str], tc: dict) -> None:
        tc_index = tc.get("index", 0)
        fn = tc.get("function") or {}
        name = fn.get("name") if isinstance(fn, dict) else None
        args = (fn.get("arguments") or "") if isinstance(fn, dict) else ""
        tool_id = tc.get("id")

        # Continuation of an already-started tool call
        if tc_index in self.started_tools:
            if tc_index != self.open_tc_index:
                # Provider interleaved tool-call chunks (rare, non-sequential).
                # Re-open a continuation block so the Anthropic protocol
                # ordering (no deltas after stop) is never violated.
                logger.warning(
                    "Interleaved tool-call chunks for index %s; "
                    "opening continuation block", tc_index,
                )
                self._open(events, "tool", {
                    "type": "tool_use",
                    "id": tool_id or f"toolu_prism_{uuid4().hex[:12]}",
                    "name": name or "",
                    "input": {},
                }, tc_index=tc_index)
            if args:
                events.append(_sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": self.open_index,
                    "delta": {"type": "input_json_delta", "partial_json": args},
                }))
                self.output_chars += len(args)
            return

        # New tool call: buffer until we have a name so content_block_start
        # always announces the real tool name (clients render it immediately).
        pending = self.pending_tools.get(tc_index)
        if pending is None:
            pending = {"id": None, "name": None, "args": ""}
            self.pending_tools[tc_index] = pending
        if tool_id and not pending["id"]:
            pending["id"] = tool_id
        if name and not pending["name"]:
            pending["name"] = name
        pending["args"] += args

        if pending["name"]:
            self._start_pending(events, tc_index)

    def _start_pending(self, events: list[str], tc_index: int) -> None:
        pending = self.pending_tools.pop(tc_index)
        # Anthropic protocol requires non-empty name in tool_use blocks.
        # Use a synthetic name if provider never sent one.
        tool_name = pending["name"] or f"unknown_tool_{tc_index}"
        self._open(events, "tool", {
            "type": "tool_use",
            "id": pending["id"] or f"toolu_prism_{uuid4().hex[:12]}",
            "name": tool_name,
            "input": {},
        }, tc_index=tc_index)
        self.started_tools.add(tc_index)
        self.saw_tool_block = True
        if pending["args"]:
            events.append(_sse("content_block_delta", {
                "type": "content_block_delta",
                "index": self.open_index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": pending["args"],
                },
            }))
            self.output_chars += len(pending["args"])

    def finalize(self, events: list[str]) -> None:
        """Flush any name-less pending tools and close the open block."""
        for tc_index in sorted(self.pending_tools.keys()):
            logger.warning(
                "Tool call %s never received a name; flushing anyway", tc_index,
            )
            self._start_pending(events, tc_index)
        self.close_open(events)


async def translate_stream(
    chunks: AsyncIterator[str],
    target_format: str,
    model_name: str,
    source_format: str = "openai-compat",
) -> AsyncIterator[str]:
    """Translate OpenAI-compatible SSE chunks to Anthropic SSE format."""
    if target_format != "anthropic":
        async for line in chunks:
            yield line
        return

    message_id = f"msg_prism_{uuid4().hex[:24]}"

    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model_name,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        },
    })
    yield _sse("ping", {"type": "ping"})

    builder = _AnthropicStreamBuilder()
    provider_usage: dict = {}
    last_finish_reason: Optional[str] = None

    async for line in chunks:
        if not line or not line.strip() or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            logger.warning(f"Skipping invalid JSON in SSE chunk: {data_str[:120]}")
            continue
        if not isinstance(data, dict):
            continue

        # Mid-stream provider errors (some providers send them as data chunks)
        if data.get("error"):
            events: list[str] = []
            builder.finalize(events)
            for e in events:
                yield e
            yield _sse("error", translate_error(data, 500))
            return

        # Capture real usage whenever a provider includes it (final chunk
        # with stream_options.include_usage, or Mistral/DeepSeek defaults).
        if isinstance(data.get("usage"), dict):
            for k, v in data["usage"].items():
                if v is not None:
                    provider_usage[k] = v

        choices = data.get("choices") or []
        if not choices:
            continue
        choice = choices[0] or {}
        delta = choice.get("delta") or {}
        events = []

        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            builder.thinking(events, str(reasoning))

        content = delta.get("content")
        if content:
            builder.text(events, str(content))

        for tc in delta.get("tool_calls") or []:
            if isinstance(tc, dict):
                builder.tool_call(events, tc)

        if choice.get("finish_reason"):
            last_finish_reason = choice["finish_reason"]

        for e in events:
            yield e

    # Close any remaining blocks (including buffered name-less tools)
    events = []
    builder.finalize(events)
    for e in events:
        yield e

    # Stop reason: emitted tool blocks always imply tool_use so agentic
    # clients execute the tools, even if the provider said plain 'stop'.
    if builder.saw_tool_block and last_finish_reason in (
        None, "stop", "tool_calls", "function_call",
    ):
        stop_reason = "tool_use"
    elif last_finish_reason:
        stop_reason = FINISH_REASON_MAP.get(last_finish_reason, "end_turn")
    else:
        stop_reason = "end_turn"

    usage = translate_usage(provider_usage)
    if not provider_usage:
        # No usage from provider: rough char-based estimate (~4 chars/token)
        usage["output_tokens"] = max(1, builder.output_chars // 4)

    # Ensure cache token fields are always present — Claude Code's SDK
    # reads these and will crash or produce wrong accounting if missing.
    usage.setdefault("cache_creation_input_tokens", 0)
    usage.setdefault("cache_read_input_tokens", 0)

    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": usage,
    })
    yield _sse("message_stop", {"type": "message_stop"})
