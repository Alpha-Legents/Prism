"""Prism proxy server — with real streaming."""

import logging
import json
import time
import asyncio
import hashlib
import httpx

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from .bridge              import get_bridge
from .probe.client        import learn_from_request
from .probe.capabilities  import learn_from_chunk
from .translate.request   import translate_request
from .translate.response  import translate_response
from .translate.headers   import translate_headers
from .translate.stream    import translate_stream

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("prism.proxy")

app = FastAPI(title="Prism", version="0.3.0")

# ── Deduplication ─────────────────────────────────────────────────────────────
_cache:    dict[str, dict]         = {}
_inflight: dict[str, asyncio.Event] = {}
_CACHE_TTL = 10

# ── Throttle ──────────────────────────────────────────────────────────────────
_throttle      = asyncio.Semaphore(1)
_REQUEST_DELAY = 2.0


def _fingerprint(body: dict) -> str:
    msgs  = body.get("messages", [])
    last  = json.dumps(msgs[-1]) if msgs else ""
    return hashlib.md5(f"{len(msgs)}:{last}".encode()).hexdigest()


# ── Compat endpoints ──────────────────────────────────────────────────────────

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    MODELS = [
        "claude-opus-4-7", "claude-opus-4-5",
        "claude-sonnet-4-6", "claude-sonnet-4-5",
        "claude-haiku-4-5", "claude-haiku-3-5",
        "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
    ]
    return {"object": "list", "data": [
        {"id": m, "object": "model", "created": 1700000000, "owned_by": "anthropic"}
        for m in MODELS
    ]}


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    try:
        body = await request.json()
        text = " ".join(
            m.get("content", "") if isinstance(m.get("content"), str) else ""
            for m in body.get("messages", [])
        )
        est = max(1, len(text) // 4)
    except Exception:
        est = 100
    return {"input_tokens": est}


@app.get("/health")
async def health():
    return get_bridge().status()


# ── Non-streaming fallback parser ─────────────────────────────────────────────

def _parse_sse_to_dict(text: str) -> dict:
    chunks, content = [], ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:") and line != "data: [DONE]":
            try: chunks.append(json.loads(line[5:].strip()))
            except: pass
    if not chunks:
        return {"error": "empty SSE stream"}
    base = chunks[-1]
    for c in chunks:
        content += (c.get("choices", [{}])[0].get("delta") or {}).get("content") or ""
    if base.get("choices"):
        msg = base["choices"][0].get("message") or base["choices"][0].get("delta", {})
        if content: msg["content"] = content
        base["choices"][0]["message"] = msg
    return base


# ── Main handler ──────────────────────────────────────────────────────────────

async def _handle(request: Request, path: str):
    bridge = get_bridge()

    if not bridge.is_configured():
        raise HTTPException(503, detail={
            "error": "Bridge not configured",
            "hint":  "Run: prism --provider URL --key KEY --model MODEL",
        })

    try:
        body: dict = await request.json()
    except Exception:
        body = {}

    req_headers   = dict(request.headers)
    client_wants_stream = body.get("stream", False)

    logger.info("")
    logger.info(f"━━━━ /{path} ━━━━")
    logger.info(f"IN  model={body.get('model')} msgs={len(body.get('messages', []))} tools={len(body.get('tools', []))} stream={client_wants_stream}")

    # ── Deduplication (skip for streaming — can't cache a stream) ─────────────
    fp = _fingerprint(body)

    if not client_wants_stream:
        now = time.time()
        for k in list(_cache.keys()):
            if now - _cache[k]["ts"] > _CACHE_TTL:
                del _cache[k]

        if fp in _cache:
            logger.info("DEDUP hit")
            cached = _cache[fp]
            return JSONResponse(content=cached["response"], status_code=200, headers=cached["headers"])

        if fp in _inflight:
            logger.info("DEDUP waiting...")
            try:
                await asyncio.wait_for(_inflight[fp].wait(), timeout=30)
            except asyncio.TimeoutError:
                pass
            if fp in _cache:
                cached = _cache[fp]
                return JSONResponse(content=cached["response"], status_code=200, headers=cached["headers"])

        _inflight[fp] = asyncio.Event()

    # ── Learn client ──────────────────────────────────────────────────────────
    if not bridge.client or not bridge.client.learned:
        bridge.client = learn_from_request(body, req_headers)
        logger.info(f"CLIENT: {bridge.client.format} / {bridge.client.tool_hint}")

    client_format   = bridge.client.format
    provider_format = bridge.provider_format

    # Resolve which provider model to use for this request
    requested_model  = body.get("model")
    resolved_model   = bridge.resolve_model(requested_model)

    if not resolved_model:
        raise HTTPException(503, detail={
            "error": f"No model mapped for '{requested_model}'",
            "hint":  "Add a mapping with --model-map or set a --model fallback",
        })

    logger.info(f"MODEL: {requested_model} → {resolved_model}")

    # ── Translate request ─────────────────────────────────────────────────────
    translated_req = translate_request(
        body,
        client_format   = client_format,
        provider_format = provider_format,
        model_override  = resolved_model,
    )
    # Always stream from provider — we handle both cases
    translated_req["stream"] = True

    out_headers = {
        "Content-Type":    "application/json",
        "accept-encoding": "identity",
    }
    if bridge.api_key:
        out_headers["Authorization"] = f"Bearer {bridge.api_key}"

    logger.info(f"→ {bridge.completion_url} [{resolved_model}] streaming=True")

    # ── Streaming response path ───────────────────────────────────────────────
    if client_wants_stream:
        async def _stream_generator():
            async with _throttle:
                async with httpx.AsyncClient(timeout=300) as client:
                    async with client.stream(
                        "POST",
                        bridge.completion_url,
                        json=translated_req,
                        headers=out_headers,
                    ) as resp:
                        if resp.status_code != 200:
                            error_body = await resp.aread()
                            logger.error(f"Provider error {resp.status_code}: {error_body[:200]}")
                            yield f"data: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': f'Provider returned {resp.status_code}'}})}\n\n"
                            return

                        async def chunk_iter():
                            async for line in resp.aiter_lines():
                                if line:
                                    yield line.encode()

                        async for event in translate_stream(
                            chunk_iter(),
                            client_format = client_format,
                            model         = bridge.model,
                        ):
                            yield event

                await asyncio.sleep(_REQUEST_DELAY)

        return StreamingResponse(
            _stream_generator(),
            media_type = "text/event-stream",
            headers    = {
                "cache-control":    "no-cache",
                "x-accel-buffering": "no",
                "anthropic-version": "2023-06-01",
                "content-encoding":  "identity",
            },
        )

    # ── Non-streaming response path ───────────────────────────────────────────
    # Still request stream=True from provider, collect all chunks, return JSON
    collected_lines: list[str] = []

    async with _throttle:
        async with httpx.AsyncClient(timeout=300) as client:
            try:
                async with client.stream(
                    "POST",
                    bridge.completion_url,
                    json=translated_req,
                    headers=out_headers,
                ) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        if fp in _inflight:
                            _inflight[fp].set()
                            del _inflight[fp]
                        try:
                            err = json.loads(error_body)
                        except Exception:
                            err = {"error": error_body.decode()[:500]}
                        return JSONResponse(content=err, status_code=resp.status_code)

                    async for line in resp.aiter_lines():
                        if line:
                            collected_lines.append(line)

            except httpx.ReadTimeout:
                logger.error("TIMEOUT")
                if fp in _inflight:
                    _inflight[fp].set()
                    del _inflight[fp]
                raise HTTPException(504, "Provider timed out")
            except httpx.ConnectError as e:
                logger.error(f"CONNECT ERROR: {e}")
                if fp in _inflight:
                    _inflight[fp].set()
                    del _inflight[fp]
                raise HTTPException(502, str(e))

        await asyncio.sleep(_REQUEST_DELAY)

    # Reassemble SSE lines into a full response dict
    raw_text = "\n".join(collected_lines)
    try:
        # Try as plain JSON first (some providers ignore stream=True)
        raw = json.loads(raw_text)
    except Exception:
        raw = _parse_sse_to_dict(raw_text)

    # Learn capabilities from full response
    learn_from_chunk(bridge.model, raw)

    logger.info(f"RAW: {json.dumps(raw)[:400]}")

    translated   = translate_response(raw, client_format)
    resp_headers = translate_headers({}, client_format)

    stop = translated.get("stop_reason") or "?"
    logger.info(f"OUT stop={stop} blocks={len(translated.get('content', []))}")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━")

    _cache[fp] = {"response": translated, "headers": resp_headers, "ts": time.time()}
    if fp in _inflight:
        _inflight[fp].set()
        del _inflight[fp]

    return JSONResponse(content=translated, status_code=200, headers=resp_headers)


@app.post("/v1/messages")
async def messages_endpoint(request: Request):
    return await _handle(request, "v1/messages")


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(request: Request, path: str):
    return await _handle(request, path)