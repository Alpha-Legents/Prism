"""
Prism proxy server - v0.5.0
"""

import json
import logging
import os
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from .core.bridge import get_bridge
from .translate.stream import translate_stream, FINISH_REASON_MAP
from .translate.errors import translate_error
from .providers.openai_compat import OpenAICompatPlugin, translate_usage

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s"
)
logger = logging.getLogger("prism.proxy")

app = FastAPI(title="Prism", version="0.5.0")

# Status codes that indicate the current API key is exhausted/blocked and we
# should rotate to the next one. Transient 5xx errors keep the current key.
KEY_ROTATION_STATUS_CODES = {401, 402, 403, 429}

# ── Health ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return get_bridge().status()


# ── Model list (kept for compatibility) ────────────────────────────────────

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    bridge = get_bridge()
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": 1700000000, "owned_by": "prism"}
            for m in (list(bridge.config.model_map.keys()) if bridge.config.model_map else [])
            or ([bridge.config.model] if bridge.config.model else [])
        ],
    }


# ── Token counting (improved) ──────────────────────────────────────────────

@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    try:
        body = await request.json()
        # Character-based heuristic covering text, thinking, tool_use inputs
        # and tool_result contents (agentic transcripts are tool-heavy, so
        # counting only text blocks badly underestimates).
        text_parts = []
        for m in body.get("messages", []):
            content = m.get("content", "")
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "text":
                        text_parts.append(block.get("text", ""))
                    elif btype == "thinking":
                        text_parts.append(block.get("thinking", ""))
                    elif btype == "tool_use":
                        try:
                            text_parts.append(json.dumps(block.get("input", {})))
                        except (TypeError, ValueError):
                            pass
                    elif btype == "tool_result":
                        rc = block.get("content", "")
                        if isinstance(rc, str):
                            text_parts.append(rc)
                        elif isinstance(rc, list):
                            for item in rc:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    text_parts.append(item.get("text", ""))

        total_chars = sum(len(p) for p in text_parts)
        # Rough estimate: ~4 chars per token for English text
        est = max(1, total_chars // 4)

        # Add overhead for tools and system
        tools = body.get("tools", [])
        if tools:
            est += len(json.dumps(tools)) // 4
        if body.get("system"):
            est += len(str(body["system"])) // 4

    except Exception:
        est = 100

    return {"input_tokens": est}


# ── Main message handlers ─────────────────────────────────────────────────

@app.post("/v1/messages")
@app.post("/messages")
async def messages_endpoint(request: Request):
    return await _handle_request(request, "messages")


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions_endpoint(request: Request):
    return await _handle_request(request, "chat/completions")


async def _handle_request(request: Request, path: str):
    """Main request handler with proper translation."""
    bridge = get_bridge()

    if not bridge.is_configured():
        raise HTTPException(503, detail={
            "error": "Bridge not configured",
            "hint": "Run: prism --provider URL --key KEY --model MODEL",
        })

    try:
        body = await request.json()
    except Exception:
        body = {}

    # Extract and preserve beta headers from the client
    beta_headers = _extract_beta_headers(request.headers)

    client_wants_stream = body.get("stream", False)
    requested_model = body.get("model")
    resolved_model = bridge.resolve_model(requested_model)

    if not resolved_model:
        raise HTTPException(503, detail={
            "error": f"No model mapped for '{requested_model}'",
            "hint": "Use --model-map or --model to configure model mapping",
        })

    # Use the provider plugin to translate request
    plugin = OpenAICompatPlugin()

    # Critical: model_override must be the RESOLVED model, not the raw model
    translated = plugin.translate_request(body, model_override=resolved_model)
    translated["model"] = resolved_model  # Force the resolved model

    # ALWAYS stream from provider internally
    translated["stream"] = True

    # Opt-in: ask OpenAI-compatible providers for real token usage in the
    # final stream chunk. Off by default because some strict providers
    # (e.g. Mistral) reject unknown params; providers like Mistral/DeepSeek
    # include usage in the final chunk anyway and it is captured either way.
    if os.environ.get("PRISM_STREAM_USAGE", "").lower() in ("1", "true", "yes"):
        translated["stream_options"] = {"include_usage": True}

    # Get API key
    api_key = bridge.get_current_api_key()
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Accept-Encoding": "identity",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Preserve beta headers from client (ignored by OpenAI-compat providers,
    # meaningful for Anthropic-native passthrough providers)
    if beta_headers:
        headers["anthropic-beta"] = ",".join(beta_headers)
        logger.debug(f"Preserving beta headers: {beta_headers}")

    provider_url = bridge.config.provider_url
    if not provider_url.endswith("/chat/completions"):
        provider_url = provider_url.rstrip("/") + "/chat/completions"

    logger.info(f"→ {provider_url} [{resolved_model}] stream={client_wants_stream}")

    if client_wants_stream:
        return await _stream_response(translated, headers, provider_url, resolved_model, bridge)
    else:
        return await _non_stream_response(translated, headers, provider_url, resolved_model, bridge, body)


def _extract_beta_headers(headers) -> list:
    """Extract anthropic-beta headers from client request."""
    beta_str = headers.get("anthropic-beta") or headers.get("Anthropic-Beta")
    if beta_str:
        return [b.strip() for b in beta_str.split(",") if b.strip()]
    return []


def _rotate_key_if_needed(bridge, status_code: int) -> None:
    """Rotate API keys only for auth/quota/rate-limit errors.

    Rotating on transient 5xx errors would needlessly mark healthy keys as
    exhausted and churn through the pool.
    """
    if status_code in KEY_ROTATION_STATUS_CODES:
        bridge.mark_key_exhausted(bridge.config.key_index)
        bridge.advance_key()


async def _stream_response(translated, headers, provider_url, resolved_model, bridge):
    """Handle streaming response with proper SSE translation."""

    async def _stream_gen():
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream("POST", provider_url, json=translated, headers=headers) as resp:
                if resp.status_code != 200:
                    err_body = await resp.aread()
                    logger.error(f"Provider error: {resp.status_code} {err_body[:200]}")
                    _rotate_key_if_needed(bridge, resp.status_code)
                    err_text = err_body.decode("utf-8", errors="replace")
                    try:
                        err_json = json.loads(err_text)
                    except json.JSONDecodeError:
                        err_json = err_text
                    anthro_error = translate_error(err_json, resp.status_code)
                    yield f"event: error\ndata: {json.dumps(anthro_error)}\n\n"
                    return

                # Advance key on success for round-robin (parity with the
                # non-streaming path)
                bridge.advance_key()

                # Feed chunks to translator
                async for line in translate_stream(resp.aiter_lines(), "anthropic", resolved_model, source_format="openai-compat"):
                    yield line

    # Translate response headers back to Anthropic format
    response_headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "anthropic-version": "2023-06-01",
        "request-id": f"req_prism_{uuid4().hex[:16]}",
    }

    return StreamingResponse(
        _stream_gen(),
        media_type="text/event-stream",
        headers=response_headers,
    )


async def _non_stream_response(translated, headers, provider_url, resolved_model, bridge, original_body):
    """Handle non-streaming response by streaming from provider and buffering."""

    # Stream from provider, collecting text, reasoning, tool_calls, usage and
    # finish_reason, then assemble a single Anthropic message.
    collected_text = []
    collected_reasoning = []
    tool_calls_by_index: dict = {}
    finish_reason = None
    provider_usage: dict = {}
    response_model = resolved_model

    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream("POST", provider_url, json=translated, headers=headers) as resp:
            if resp.status_code != 200:
                err_body = await resp.aread()
                logger.error(f"Provider error: {resp.status_code} {err_body[:200]}")
                _rotate_key_if_needed(bridge, resp.status_code)
                err_text = err_body.decode("utf-8", errors="replace")
                try:
                    err_json = json.loads(err_text)
                except json.JSONDecodeError:
                    err_json = err_text
                anthro_error = translate_error(err_json, resp.status_code)
                return JSONResponse(
                    content=anthro_error,
                    status_code=resp.status_code,
                )

            # Advance key on success for round-robin
            bridge.advance_key()

            # Stream and collect content
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue

                # Mid-stream provider errors
                if data.get("error"):
                    anthro_error = translate_error(data, 502)
                    return JSONResponse(content=anthro_error, status_code=502)

                # Capture real usage whenever a provider includes it
                if isinstance(data.get("usage"), dict):
                    for k, v in data["usage"].items():
                        if v is not None:
                            provider_usage[k] = v
                if data.get("model"):
                    response_model = data["model"]

                choices = data.get("choices", [])
                if choices:
                    choice_data = choices[0]
                    delta = choice_data.get("delta", {})

                    # Collect reasoning (DeepSeek/OpenRouter reasoning models)
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning:
                        collected_reasoning.append(str(reasoning))

                    # Collect text content
                    content = delta.get("content", "")
                    if content:
                        collected_text.append(content)

                    # Collect tool_calls
                    delta_tool_calls = delta.get("tool_calls", []) or []
                    for tc in delta_tool_calls:
                        index = tc.get("index", 0)
                        fn = tc.get("function", {}) or {}
                        if index not in tool_calls_by_index:
                            tool_calls_by_index[index] = {
                                "id": tc.get("id", ""),
                                "type": tc.get("type", "function"),
                                "function": {
                                    "name": fn.get("name", ""),
                                    "arguments": fn.get("arguments", "") or "",
                                }
                            }
                        else:
                            existing = tool_calls_by_index[index]
                            if tc.get("id") and not existing["id"]:
                                existing["id"] = tc["id"]
                            if fn.get("name") and not existing["function"]["name"]:
                                existing["function"]["name"] = fn["name"]
                            existing["function"]["arguments"] += fn.get("arguments", "") or ""

                    # Capture finish_reason if present
                    if choice_data.get("finish_reason"):
                        finish_reason = choice_data["finish_reason"]

    # Build the Anthropic response directly
    content_blocks: list = []

    reasoning_text = "".join(collected_reasoning)
    if reasoning_text:
        content_blocks.append({
            "type": "thinking",
            "thinking": reasoning_text,
            "signature": "",
        })

    full_text = "".join(collected_text)
    if full_text:
        content_blocks.append({"type": "text", "text": full_text})

    for index in sorted(tool_calls_by_index.keys()):
        tc = tool_calls_by_index[index]
        raw_args = tc["function"]["arguments"]
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            args = {"_raw": raw_args}
        content_blocks.append({
            "type": "tool_use",
            "id": tc["id"] or f"toolu_prism_{uuid4().hex[:12]}",
            "name": tc["function"]["name"],
            "input": args,
        })

    # Tool calls always imply tool_use so agentic clients execute the tools
    if tool_calls_by_index:
        stop_reason = "tool_use"
    else:
        stop_reason = FINISH_REASON_MAP.get(finish_reason or "stop", "end_turn")

    usage = translate_usage(provider_usage)
    if not provider_usage:
        # No usage from provider: rough char-based estimate (~4 chars/token)
        usage["output_tokens"] = max(1, (len(full_text) + len(reasoning_text)) // 4)

    anthropic_response = {
        "id": f"msg_prism_{uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": response_model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }

    return JSONResponse(
        content=anthropic_response,
        status_code=200,
        headers={
            "anthropic-version": "2023-06-01",
            "request-id": f"req_prism_{uuid4().hex[:16]}",
        },
    )


# ── Bridge endpoints (stubs for now) ────────────────────────────────────────

@app.post("/v1/environments/bridge")
async def bridge_endpoint(request: Request):
    """Bridge endpoint for Claude Code internal communication."""
    return {"status": "ok", "bridge": "available"}


@app.get("/v1/environments/{environment_id}/work/poll")
async def poll_work(environment_id: str):
    """Poll endpoint for bridge mode."""
    # Return empty work list - this is a fallback behavior
    return {"work": []}


@app.post("/v1/environments/{environment_id}/work/{work_id}/ack")
async def ack_work(environment_id: str, work_id: str):
    """Ack endpoint for bridge mode."""
    return {"status": "acknowledged"}
