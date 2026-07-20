"""
Prism proxy server — v0.7.0

Universal LLM protocol bridge. Works with:
- Claude Code (Anthropic format)
- Cline, Roo Code (Anthropic format)
- Cursor, Aider, Continue (OpenAI format)
- Any frontend speaking Anthropic or OpenAI API

Supports: streaming, tool use, thinking/reasoning, prompt caching,
token counting, model mapping, multi-key round-robin.
"""

import json
import logging
import os
import time
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from .core.bridge import get_bridge
from .translate.stream import translate_stream, FINISH_REASON_MAP
from .translate.errors import translate_error
from .providers.openai_compat import OpenAICompatPlugin, translate_usage

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s"
)
logger = logging.getLogger("prism.proxy")

app = FastAPI(title="Prism", version="0.7.0")

# CORS support for browser-based tools (Cursor, Windsurf, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Status codes that indicate the current API key is exhausted/blocked and we
# should rotate to the next one. Transient 5xx errors keep the current key.
KEY_ROTATION_STATUS_CODES = {401, 402, 403, 429}

# Request counter for metrics
_request_count = 0
_error_count = 0
_start_time = time.time()


# ── Health & Status ──────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check with full status dashboard."""
    bridge = get_bridge()
    uptime = time.time() - _start_time
    return {
        "status": "ok" if bridge.is_configured() else "not_configured",
        "version": "0.7.0",
        "uptime_seconds": round(uptime, 1),
        "requests": _request_count,
        "errors": _error_count,
        "provider": bridge.config.provider_url or "none",
        "model": bridge.config.model or "none",
        "model_map_count": len(bridge.config.model_map) if bridge.config.model_map else 0,
        "keys_configured": len(bridge.config.api_keys) if bridge.config.api_keys else (1 if bridge.config.api_key else 0),
        "endpoints": {
            "anthropic": ["/v1/messages", "/messages"],
            "openai": ["/v1/chat/completions", "/chat/completions"],
            "models": ["/v1/models", "/models"],
            "token_count": ["/v1/messages/count_tokens"],
        },
    }


@app.get("/")
async def root():
    """Root endpoint with setup instructions."""
    bridge = get_bridge()
    if not bridge.is_configured():
        return {
            "message": "Prism is running but not configured.",
            "setup": {
                "quick_start": "prism --provider <URL> --key <KEY> --model <MODEL>",
                "interactive": "prism --provider <URL> --key <KEY> --interactive",
                "examples": [
                    "prism --provider https://openrouter.ai/api/v1 --key sk-... --interactive",
                    "prism --provider https://api.groq.com/openai/v1 --key sk-... --model llama-3.3-70b-versatile",
                ],
            },
        }
    return {
        "message": "Prism is running",
        "provider": bridge.config.provider_url,
        "model": bridge.config.model,
        "status": "ready",
    }


# ── Model list ───────────────────────────────────────────────────────────

@app.get("/v1/models")
@app.get("/models")
async def list_models(request: Request):
    """List available models in OpenAI format (universal)."""
    bridge = get_bridge()

    # Get the models that are mapped
    if bridge.config.model_map:
        models = list(bridge.config.model_map.keys())
    elif bridge.config.model:
        models = [bridge.config.model]
    else:
        models = ["claude-sonnet-4-6"]  # Default model

    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": 1700000000, "owned_by": "prism"}
            for m in models
        ],
    }


# ── Token counting ───────────────────────────────────────────────────────

@app.post("/v1/messages/count_tokens")
@app.post("/messages/count_tokens")
async def count_tokens(request: Request):
    """Count tokens in a message payload.

    Returns Anthropic-compatible token count response.
    Uses character-based heuristic (~4 chars/token) for estimation.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

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
    est = max(1, total_chars // 4)

    # Add overhead for tools and system
    tools = body.get("tools", [])
    if tools:
        est += len(json.dumps(tools)) // 4
    if body.get("system"):
        system = body["system"]
        if isinstance(system, list):
            est += sum(len(str(b)) for b in system) // 4
        else:
            est += len(str(system)) // 4

    return {"input_tokens": est}


# ── Main message handlers ────────────────────────────────────────────────

@app.post("/v1/messages")
@app.post("/messages")
async def messages_endpoint(request: Request):
    return await _handle_request(request, "messages")


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions_endpoint(request: Request):
    return await _handle_request(request, "chat/completions")


# Some tools (Copilot, custom scripts) use these alternate paths
@app.post("/api/v1/chat/completions")
async def api_v1_chat_completions(request: Request):
    return await _handle_request(request, "chat/completions")


@app.post("/v1/engines/{engine_id}/chat/completions")
async def engine_chat_completions(request: Request, engine_id: str):
    return await _handle_request(request, "chat/completions")


# Catch-all for any other POST to unknown paths — try to handle it
@app.post("/{path:path}")
async def catch_all_post(request: Request, path: str):
    """Catch-all for unknown POST endpoints.

    Some tools use non-standard paths. We try to detect the format
    and route accordingly.
    """
    # Skip known non-completion paths
    if any(x in path for x in ["health", "docs", "redoc", "openapi", "environments"]):
        raise HTTPException(404, detail={"error": "Not found"})

    try:
        body = await request.json()
    except Exception:
        body = {}

    # If it looks like a completion request, handle it
    if "messages" in body or "model" in body:
        logger.debug(f"Catch-all routing {path} as completion request")
        return await _handle_request(request, "chat/completions")

    raise HTTPException(404, detail={"error": "Not found", "path": path})


async def _handle_request(request: Request, path: str):
    """Main request handler with proper translation.

    Auto-detects incoming format (Anthropic or OpenAI) and translates
    accordingly. Works with Claude Code, Cline, Cursor, Aider, Continue,
    and any other frontend that speaks either protocol.
    """
    global _request_count, _error_count
    _request_count += 1

    bridge = get_bridge()

    if not bridge.is_configured():
        _error_count += 1
        raise HTTPException(503, detail={
            "error": "Bridge not configured",
            "hint": "Run: prism --provider URL --key KEY --model MODEL",
        })

    try:
        body = await request.json()
    except Exception:
        body = {}

    # Auto-detect incoming format from the request body
    client_format = _detect_client_format(body, path)
    logger.debug(f"Detected client format: {client_format} (path={path})")

    # Extract and preserve beta headers from the client
    beta_headers = _extract_beta_headers(request.headers)

    client_wants_stream = body.get("stream", False)
    requested_model = body.get("model")
    resolved_model = bridge.resolve_model(requested_model)

    if not resolved_model:
        _error_count += 1
        raise HTTPException(503, detail={
            "error": f"No model mapped for '{requested_model}'",
            "hint": "Use --model-map, --model, or --interactive to configure model mapping",
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

    # Get provider URL first — needed for beta header check
    provider_url = bridge.config.provider_url
    if not provider_url.endswith("/chat/completions"):
        provider_url = provider_url.rstrip("/") + "/chat/completions"

    # Strip beta headers for OpenAI-compat providers — they don't understand
    # Anthropic beta headers and may reject them as unknown headers.
    # Only preserve for Anthropic-native passthrough providers.
    if beta_headers:
        # Check if provider is Anthropic-native (not OpenAI-compat)
        is_anthropic_native = "anthropic" in provider_url.lower() and "openai" not in provider_url.lower()
        if is_anthropic_native:
            headers["anthropic-beta"] = ",".join(beta_headers)
            logger.debug(f"Preserving beta headers for Anthropic-native: {beta_headers}")
        else:
            logger.debug(f"Stripping {len(beta_headers)} beta headers for OpenAI-compat provider")

    logger.info(f"→ {provider_url} [{resolved_model}] stream={client_wants_stream} format={client_format}")

    if client_wants_stream:
        return await _stream_response(translated, headers, provider_url, resolved_model, bridge, client_format)
    else:
        return await _non_stream_response(translated, headers, provider_url, resolved_model, bridge, body, client_format)


def _extract_beta_headers(headers) -> list:
    """Extract anthropic-beta headers from client request."""
    beta_str = headers.get("anthropic-beta") or headers.get("Anthropic-Beta")
    if beta_str:
        return [b.strip() for b in beta_str.split(",") if b.strip()]
    return []


def _detect_client_format(body: dict, path: str) -> str:
    """Auto-detect the client's API format from the request body and path.

    Returns 'anthropic' or 'openai-compat'.
    """
    # Path-based detection is most reliable
    if "messages" in path and "chat" not in path:
        return "anthropic"
    if "chat/completions" in path:
        return "openai-compat"

    # Body-based detection
    if not body:
        return "anthropic"  # Default to Anthropic for empty bodies

    # Anthropic indicators
    if "system" in body:
        return "anthropic"
    if "betas" in body:
        return "anthropic"
    if "max_tokens" in body and "messages" in body:
        # Check if messages have Anthropic-style content blocks
        messages = body.get("messages", [])
        if messages and isinstance(messages[0], dict):
            content = messages[0].get("content")
            if isinstance(content, list) and content and isinstance(content[0], dict):
                if "type" in content[0]:
                    return "anthropic"

    # OpenAI indicators
    if "messages" in body:
        messages = body.get("messages", [])
        if messages and isinstance(messages[0], dict):
            content = messages[0].get("content")
            if isinstance(content, str):
                return "openai-compat"
            # Check for tool_calls in assistant messages
            if messages[0].get("role") == "assistant" and "tool_calls" in messages[0]:
                return "openai-compat"

    # Default to Anthropic (Claude Code is the primary target)
    return "anthropic"


def _rotate_key_if_needed(bridge, status_code: int) -> None:
    """Rotate API keys only for auth/quota/rate-limit errors."""
    if status_code in KEY_ROTATION_STATUS_CODES:
        bridge.mark_key_exhausted(bridge.config.key_index)
        bridge.advance_key()


# ── Streaming response ───────────────────────────────────────────────────

async def _stream_response(translated, headers, provider_url, resolved_model, bridge, client_format="anthropic"):
    """Handle streaming response with proper SSE translation."""

    async def _stream_gen():
        global _error_count
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream("POST", provider_url, json=translated, headers=headers) as resp:
                    if resp.status_code != 200:
                        err_body = await resp.aread()
                        _error_count += 1
                        logger.error(f"Provider error: {resp.status_code} {err_body[:200]}")
                        _rotate_key_if_needed(bridge, resp.status_code)
                        err_text = err_body.decode("utf-8", errors="replace")
                        try:
                            err_json = json.loads(err_text)
                        except json.JSONDecodeError:
                            err_json = err_text
                        retry_after = None
                        retry_header = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
                        if retry_header:
                            try:
                                retry_after = float(retry_header)
                            except (ValueError, TypeError):
                                pass
                        anthro_error = translate_error(err_json, resp.status_code, retry_after=retry_after)
                        if client_format == "anthropic":
                            yield f"event: error\ndata: {json.dumps(anthro_error)}\n\n"
                        else:
                            yield f"data: {json.dumps({'error': anthro_error['error']})}\n\n"
                            yield "data: [DONE]\n\n"
                        return

                    bridge.advance_key()

                    if client_format == "anthropic":
                        async for line in translate_stream(resp.aiter_lines(), "anthropic", resolved_model, source_format="openai-compat"):
                            yield line
                    else:
                        async for line in resp.aiter_lines():
                            if line.strip():
                                yield line + "\n" if not line.endswith("\n") else line
        except httpx.TimeoutException:
            _error_count += 1
            logger.error(f"Provider timeout: {provider_url}")
            if client_format == "anthropic":
                anthro_error = translate_error({"error": {"message": "Provider request timed out"}}, 504)
                yield f"event: error\ndata: {json.dumps(anthro_error)}\n\n"
            else:
                yield f"data: {json.dumps({'error': {'message': 'Provider request timed out', 'type': 'timeout_error'}})}\n\n"
                yield "data: [DONE]\n\n"
        except httpx.ConnectError as e:
            _error_count += 1
            logger.error(f"Provider connection error: {e}")
            if client_format == "anthropic":
                anthro_error = translate_error({"error": {"message": f"Failed to connect to provider: {e}"}}, 502)
                yield f"event: error\ndata: {json.dumps(anthro_error)}\n\n"
            else:
                yield f"data: {json.dumps({'error': {'message': f'Failed to connect to provider: {e}', 'type': 'connection_error'}})}\n\n"
                yield "data: [DONE]\n\n"
        except Exception as e:
            _error_count += 1
            logger.error(f"Unexpected error in stream: {e}")
            if client_format == "anthropic":
                anthro_error = translate_error({"error": {"message": f"Internal proxy error: {e}"}}, 500)
                yield f"event: error\ndata: {json.dumps(anthro_error)}\n\n"
            else:
                yield f"data: {json.dumps({'error': {'message': f'Internal proxy error: {e}', 'type': 'server_error'}})}\n\n"
                yield "data: [DONE]\n\n"

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


# ── Non-streaming response ──────────────────────────────────────────────

async def _non_stream_response(translated, headers, provider_url, resolved_model, bridge, original_body, client_format="anthropic"):
    """Handle non-streaming response by streaming from provider and buffering."""
    global _error_count

    collected_text = []
    collected_reasoning = []
    tool_calls_by_index: dict = {}
    finish_reason = None
    provider_usage: dict = {}
    response_model = resolved_model

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream("POST", provider_url, json=translated, headers=headers) as resp:
                if resp.status_code != 200:
                    _error_count += 1
                    err_body = await resp.aread()
                    logger.error(f"Provider error: {resp.status_code} {err_body[:200]}")
                    _rotate_key_if_needed(bridge, resp.status_code)
                    err_text = err_body.decode("utf-8", errors="replace")
                    try:
                        err_json = json.loads(err_text)
                    except json.JSONDecodeError:
                        err_json = err_text
                    retry_after = None
                    retry_header = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
                    if retry_header:
                        try:
                            retry_after = float(retry_header)
                        except (ValueError, TypeError):
                            pass
                    anthro_error = translate_error(err_json, resp.status_code, retry_after=retry_after)
                    return JSONResponse(content=anthro_error, status_code=resp.status_code)

                bridge.advance_key()

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

                    if data.get("error"):
                        anthro_error = translate_error(data, 502)
                        return JSONResponse(content=anthro_error, status_code=502)

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

                        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                        if reasoning:
                            collected_reasoning.append(str(reasoning))

                        content = delta.get("content", "")
                        if content:
                            collected_text.append(content)

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

                        if choice_data.get("finish_reason"):
                            finish_reason = choice_data["finish_reason"]
    except httpx.TimeoutException:
        _error_count += 1
        logger.error(f"Provider timeout: {provider_url}")
        anthro_error = translate_error({"error": {"message": "Provider request timed out"}}, 504)
        return JSONResponse(content=anthro_error, status_code=504)
    except httpx.ConnectError as e:
        _error_count += 1
        logger.error(f"Provider connection error: {e}")
        anthro_error = translate_error({"error": {"message": f"Failed to connect to provider: {e}"}}, 502)
        return JSONResponse(content=anthro_error, status_code=502)
    except Exception as e:
        _error_count += 1
        logger.error(f"Unexpected error in non-streaming: {e}")
        anthro_error = translate_error({"error": {"message": f"Internal proxy error: {e}"}}, 500)
        return JSONResponse(content=anthro_error, status_code=500)

    # Build response in the format the client expects
    if client_format == "anthropic":
        return _build_anthropic_response(
            collected_text, collected_reasoning, tool_calls_by_index,
            finish_reason, provider_usage, response_model
        )
    else:
        return _build_openai_response(
            collected_text, collected_reasoning, tool_calls_by_index,
            finish_reason, provider_usage, response_model
        )


def _build_anthropic_response(collected_text, collected_reasoning, tool_calls_by_index,
                               finish_reason, provider_usage, response_model):
    """Build Anthropic-format response."""
    content_blocks = []

    reasoning_text = "".join(collected_reasoning)
    if reasoning_text:
        content_blocks.append({"type": "thinking", "thinking": reasoning_text, "signature": ""})

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

    if tool_calls_by_index:
        stop_reason = "tool_use"
    else:
        stop_reason = FINISH_REASON_MAP.get(finish_reason or "stop", "end_turn")

    usage = translate_usage(provider_usage)
    if not provider_usage:
        usage["output_tokens"] = max(1, (len(full_text) + len(reasoning_text)) // 4)
    usage.setdefault("cache_creation_input_tokens", 0)
    usage.setdefault("cache_read_input_tokens", 0)

    response = {
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
        content=response,
        status_code=200,
        headers={"anthropic-version": "2023-06-01", "request-id": f"req_prism_{uuid4().hex[:16]}"},
    )


def _build_openai_response(collected_text, collected_reasoning, tool_calls_by_index,
                            finish_reason, provider_usage, response_model):
    """Build OpenAI-format response."""
    reasoning_text = "".join(collected_reasoning)
    full_text = "".join(collected_text)

    if tool_calls_by_index:
        openai_finish = "tool_calls"
    else:
        finish_map = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length"}
        openai_finish = finish_map.get(FINISH_REASON_MAP.get(finish_reason or "stop", "end_turn"), "stop")

    message = {"role": "assistant", "content": full_text or ""}
    if reasoning_text:
        message["reasoning_content"] = reasoning_text

    openai_tool_calls = []
    for index in sorted(tool_calls_by_index.keys()):
        tc = tool_calls_by_index[index]
        openai_tool_calls.append({
            "id": tc["id"] or f"call_{uuid4().hex[:12]}",
            "type": "function",
            "function": {
                "name": tc["function"]["name"],
                "arguments": tc["function"]["arguments"] or "{}",
            },
        })
    if openai_tool_calls:
        message["tool_calls"] = openai_tool_calls

    usage = translate_usage(provider_usage)
    if not provider_usage:
        usage["output_tokens"] = max(1, (len(full_text) + len(reasoning_text)) // 4)

    response = {
        "id": f"chatcmpl-{uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": response_model,
        "choices": [{"index": 0, "message": message, "finish_reason": openai_finish}],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
    }

    return JSONResponse(content=response, status_code=200)


# ── Bridge endpoints (stubs for Claude Code internal communication) ──────

@app.post("/v1/environments/bridge")
async def bridge_endpoint(request: Request):
    return {"status": "ok", "bridge": "available"}


@app.get("/v1/environments/{environment_id}/work/poll")
async def poll_work(environment_id: str):
    return {"work": []}


@app.post("/v1/environments/{environment_id}/work/{work_id}/ack")
async def ack_work(environment_id: str, work_id: str):
    return {"status": "acknowledged"}
