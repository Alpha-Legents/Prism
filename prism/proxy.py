"""
Prism proxy server — DEBUG MODE
Full request/response logging. Explicit route handling for Claude Code.
"""

import logging
import json
import httpx

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from .bridge             import get_bridge
from .probe.client       import learn_from_request
from .translate.request  import translate_request
from .translate.response import translate_response
from .translate.headers  import translate_headers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("prism.proxy")

app = FastAPI(title="Prism", version="0.2.0")


# ── Spoof endpoints for Claude Code ──────────────────────────────────────────

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    CLAUDE_MODELS = [
        "claude-opus-4-7", "claude-opus-4-5",
        "claude-sonnet-4-6", "claude-sonnet-4-5",
        "claude-haiku-4-5", "claude-haiku-3-5",
        "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
    ]
    logger.info("[/v1/models] spoofing model list")
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": 1700000000, "owned_by": "anthropic"}
            for m in CLAUDE_MODELS
        ],
    }


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    try:
        body  = await request.json()
        msgs  = body.get("messages", [])
        text  = " ".join(
            m.get("content", "") if isinstance(m.get("content"), str) else ""
            for m in msgs
        )
        est = max(1, len(text) // 4)
    except Exception:
        est = 100
    logger.info(f"[/count_tokens] → {est}")
    return {"input_tokens": est}


@app.get("/health")
async def health():
    return get_bridge().status()


def _parse_sse(text: str) -> dict:
    """
    Extract the final complete message from an SSE stream.
    Merges all content deltas into one response object.
    """
    import json as _json
    chunks = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:") and line != "data: [DONE]":
            try:
                chunks.append(_json.loads(line[5:].strip()))
            except Exception:
                pass

    if not chunks:
        return {"error": "empty SSE stream"}

    # Use last chunk as base, merge all content deltas
    base = chunks[-1]
    content = ""
    for chunk in chunks:
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        content += delta.get("content") or ""

    # Inject merged content into base
    if base.get("choices"):
        msg = base["choices"][0].get("message") or base["choices"][0].get("delta", {})
        if content:
            msg["content"] = content
        base["choices"][0]["message"] = msg

    logger.info(f"SSE merged: {len(chunks)} chunks → {len(content)} chars")
    return base


# ── Main translation handler ──────────────────────────────────────────────────

async def _handle(request: Request, path: str):
    bridge = get_bridge()

    if not bridge.is_configured():
        logger.warning("Bridge not configured!")
        raise HTTPException(503, detail="Bridge not configured — run prism with --provider and --model")

    try:
        body: dict = await request.json()
    except Exception:
        body = {}

    req_headers = dict(request.headers)

    logger.info(f"")
    logger.info(f"━━━━ INCOMING /{path} ━━━━")
    logger.info(f"CC→PRISM model={body.get('model')} msgs={len(body.get('messages', []))} tools={len(body.get('tools', []))}")

    if not bridge.client or not bridge.client.learned:
        bridge.client = learn_from_request(body, req_headers)
        logger.info(f"CLIENT LEARNED: format={bridge.client.format} tool={bridge.client.tool_hint}")

    client_format   = bridge.client.format
    provider_format = bridge.provider.format
    if not provider_format or provider_format == "unknown":
        provider_format = "openai-compat"

    logger.info(f"TRANSLATE: {client_format} → {provider_format} | model → {bridge.model}")

    translated_req = translate_request(
        body,
        client_format=client_format,
        provider_format=provider_format,
        model_override=bridge.model,
    )
    # Force non-streaming — prism doesn't support SSE passthrough yet
    translated_req["stream"] = False

    logger.info(f"PRISM→PROVIDER: model={translated_req.get('model')} msgs={len(translated_req.get('messages', []))}")

    out_headers: dict[str, str] = {"Content-Type": "application/json"}
    if bridge.provider.api_key:
        out_headers["Authorization"] = f"Bearer {bridge.provider.api_key}"

    target_url = bridge.provider.completion_url
    logger.info(f"→ POST {target_url}")

    async with httpx.AsyncClient(timeout=300, headers={"accept-encoding": "identity"}) as client:
        try:
            resp = await client.post(target_url, json=translated_req, headers=out_headers)
        except httpx.ReadTimeout:
            logger.error("TIMEOUT — provider too slow, try a faster model")
            raise HTTPException(504, "Provider timed out")
        except httpx.ConnectError as e:
            logger.error(f"CONNECT ERROR: {e}")
            raise HTTPException(502, f"Cannot reach provider: {e}")

    logger.info(f"PROVIDER STATUS: {resp.status_code}")

    # ── Handle streaming response (SSE) ──────────────────────────────────────
    content_type = resp.headers.get("content-type", "")
    if "text/event-stream" in content_type or resp.text.strip().startswith("data:"):
        logger.info("PROVIDER returned SSE stream — extracting last complete chunk")
        raw_resp = _parse_sse(resp.text)
    else:
        try:
            raw_resp: dict = resp.json()
        except Exception:
            logger.error(f"NON-JSON from provider: {resp.text[:300]}")
            return JSONResponse(
                content={"error": "provider returned non-JSON", "body": resp.text[:500]},
                status_code=resp.status_code,
            )

    logger.info(f"PROVIDER RAW: {json.dumps(raw_resp)[:600]}")

    translated_resp    = translate_response(raw_resp, client_format)
    translated_headers = translate_headers(dict(resp.headers), client_format)

    # Must strip Content-Length — our translated body is a different size
    translated_headers.pop("content-length", None)
    translated_headers.pop("Content-Length", None)

    # Strip content-encoding — we return plain JSON, not compressed
    translated_headers.pop("content-encoding", None)
    translated_headers.pop("Content-Encoding", None)
    translated_headers["content-encoding"] = "identity"

    stop = translated_resp.get("stop_reason") or "?"
    logger.info(f"PRISM→CC: stop={stop} blocks={len(translated_resp.get('content', []))}")
    logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    return JSONResponse(
        content=translated_resp,
        status_code=200,
        headers=translated_headers,
    )


@app.post("/v1/messages")
async def messages_endpoint(request: Request):
    return await _handle(request, "v1/messages")


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(request: Request, path: str):
    logger.info(f"CATCH-ALL: {request.method} /{path}")
    return await _handle(request, path)