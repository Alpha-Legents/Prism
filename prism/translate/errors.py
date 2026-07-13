"""
Comprehensive error translation for provider errors.
Maps provider error shapes to Anthropic-format errors.
"""

import json
import logging

logger = logging.getLogger("prism.translate.errors")

# Status code -> (anthropic_error_type, default_message)
ERROR_MAP = {
    400: ("invalid_request_error", "Invalid request"),
    401: ("authentication_error", "Invalid API key"),
    402: ("billing_error", "Payment required or quota exceeded"),
    403: ("forbidden_error", "Access forbidden"),
    404: ("not_found_error", "Model not found"),
    408: ("timeout_error", "Request timeout"),
     409: ("conflict_error", "Resource conflict"),
     410: ("api_error", "Deprecated or unavailable endpoint"),
    413: ("invalid_request_error", "Request too large (context overflow)"),
    415: ("invalid_request_error", "Unsupported media type"),
    422: ("invalid_request_error", "Unprocessable entity"),
    429: ("rate_limit_error", "Rate limit exceeded"),
    431: ("invalid_request_error", "Request header fields too large"),
    500: ("api_error", "Provider server error"),
    502: ("api_error", "Bad gateway from provider"),
    503: ("api_error", "Provider service unavailable"),
    504: ("api_error", "Provider gateway timeout"),
    529: ("overloaded_error", "Provider is overloaded — retry with exponential backoff"),
}

# Anthropic error types that trigger specific client behavior
ANTHROPIC_RETRY_ERRORS = {"rate_limit_error", "overloaded_error", "api_error"}
ANTHROPIC_FATAL_ERRORS = {"authentication_error", "billing_error", "forbidden_error"}

# Patterns that indicate context window exceeded
CONTEXT_EXCEEDED_PATTERNS = [
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "token limit",
    "context too long",
    "too many tokens",
    "max tokens exceeded",
    "reduce length",
    "request too large",
]


def translate_error(error_response: dict | str, status_code: int) -> dict:
    """
    Translate a provider error into Anthropic error format.

    Args:
        error_response: The provider's error response body (dict or JSON string)
        status_code: HTTP status code

    Returns:
        Anthropic-format error dict:
        {"type": "error", "error": {"type": str, "message": str}}
    """
    # Parse string body if needed
    if isinstance(error_response, str):
        try:
            parsed = json.loads(error_response)
        except json.JSONDecodeError:
            parsed = {}
    else:
        parsed = error_response

    error_type, default_msg = ERROR_MAP.get(status_code, ("api_error", "Unknown error"))

    # Try to extract the provider's error message from various error shapes
    message = _extract_message(parsed) or default_msg

    # Check for context window exceeded (overrides other classifications)
    if _is_context_window_exceeded(parsed, message):
        error_type = "invalid_request_error"
        message = _clean_context_message(message)

    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        },
    }


def is_retryable(error_type: str) -> bool:
    """Check if an Anthropic error type should trigger a retry."""
    return error_type in ANTHROPIC_RETRY_ERRORS


def is_fatal(error_type: str) -> bool:
    """Check if an Anthropic error type is fatal (no retry possible)."""
    return error_type in ANTHROPIC_FATAL_ERRORS


def _extract_message(parsed: dict) -> str | None:
    """Extract error message from various provider error shapes."""
    # OpenAI format: {"error": {"message": "...", "type": "..."}}
    error_obj = parsed.get("error", parsed)

    if isinstance(error_obj, dict):
        msg = (
            error_obj.get("message")
            or error_obj.get("msg")
            or error_obj.get("error")
            or parsed.get("message")
        )
        if isinstance(msg, str):
            return msg.strip() or None
        if isinstance(msg, dict):
            return msg.get("message") or str(msg)

    if isinstance(error_obj, str):
        return error_obj.strip() or None

    return None


def _is_context_window_exceeded(parsed: dict, message: str) -> bool:
    """Detect context window exceeded errors from any provider."""
    error_obj = parsed.get("error", parsed)
    error_type = ""
    if isinstance(error_obj, dict):
        error_type = error_obj.get("type", "") or error_obj.get("code", "")

    msg_lower = message.lower() if message else ""

    if "context_length_exceeded" in error_type.lower():
        return True
    return any(p in msg_lower for p in CONTEXT_EXCEEDED_PATTERNS)


def _clean_context_message(message: str) -> str:
    """Clean up context window exceeded messages for the client."""
    msg_lower = message.lower()
    for p in CONTEXT_EXCEEDED_PATTERNS:
        if p in msg_lower:
            return f"Prompt too long: context window exceeded. Try shortening the conversation or summarizing context."
    return message



def extract_quota_info(error_response: dict) -> dict | None:
    """Extract quota/rate limit info from provider errors."""
    # OpenAI format
    error = error_response.get("error", {})
    if isinstance(error, dict):
        if "rate_limit" in error.get("type", "").lower():
            return {"type": "rate_limit", "retry_after": error.get("retry_after")}
        if "quota" in error.get("type", "").lower():
            return {"type": "quota_exceeded"}
    return None
