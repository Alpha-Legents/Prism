"""Header translation — maps provider headers → client-expected headers."""

STRIP = {
    "server", "via", "x-powered-by", "cf-cache-status",
    "alt-svc", "nel", "report-to", "x-envoy-upstream-service-time",
    "content-length", "content-encoding", "transfer-encoding",
}

INJECT_ANTHROPIC = {
    "anthropic-version": "2023-06-01",
    "content-type":      "application/json",
    "content-encoding":  "identity",
}

INJECT_OPENAI = {
    "content-type":     "application/json",
    "content-encoding": "identity",
}


def translate_headers(incoming: dict, client_format: str) -> dict:
    out    = {k.lower(): v for k, v in incoming.items() if k.lower() not in STRIP}
    inject = INJECT_ANTHROPIC if client_format == "anthropic" else INJECT_OPENAI
    out.update(inject)
    return out