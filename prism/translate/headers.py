"""Header translation — maps provider headers → client-expected headers."""

STRIP = {
    "server", "via", "x-powered-by", "cf-cache-status",
    "alt-svc", "nel", "report-to", "x-envoy-upstream-service-time",
}

TO_ANTHROPIC_HEADERS = {
    "x-request-id":          "request-id",
    "x-ratelimit-remaining": "anthropic-ratelimit-requests-remaining",
    "x-ratelimit-limit":     "anthropic-ratelimit-requests-limit",
}

TO_OPENAI_HEADERS = {
    "request-id":                             "x-request-id",
    "anthropic-ratelimit-requests-remaining": "x-ratelimit-remaining-requests",
}

INJECT_ANTHROPIC = {
    "anthropic-version": "2023-06-01",
    "content-type":      "application/json",
}

INJECT_OPENAI = {
    "content-type": "application/json",
}


def translate_headers(incoming: dict, client_format: str) -> dict:
    normalized = {k.lower(): v for k, v in incoming.items()}
    hmap   = TO_ANTHROPIC_HEADERS if client_format == "anthropic" else TO_OPENAI_HEADERS
    inject = INJECT_ANTHROPIC      if client_format == "anthropic" else INJECT_OPENAI

    out = {}
    for k, v in normalized.items():
        if k in STRIP:
            continue
        out[hmap.get(k, k)] = v

    out.update(inject)
    return out
