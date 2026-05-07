"""
Known frontend model lists.
These are the model IDs each client tool sends in requests.
Used for the --model-map interactive mapping flow.
"""

FRONTEND_MODELS: dict[str, list[str]] = {
    "claude-code": [
        "claude-opus-4-7",
        "claude-opus-4-5",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "claude-haiku-3-5",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
    ],
    "aider": [
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "deepseek/deepseek-coder",
        "ollama/codellama",
    ],
    "cursor": [
        "gpt-4o",
        "gpt-4-turbo",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
    ],
    "continue": [
        "gpt-4o",
        "gpt-3.5-turbo",
        "claude-3-5-sonnet-20241022",
    ],
    "opencode": [
        "gpt-4o",
        "claude-3-5-sonnet-20241022",
        "claude-sonnet-4-6",
    ],
    "cline": [
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "gpt-4o",
    ],
}

# Fallback — show generic list if tool unknown
GENERIC_MODELS = [
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
]


def get_frontend_models(tool_hint: str) -> list[str]:
    """Get known model list for a detected client tool."""
    # Normalize tool hint
    for key in FRONTEND_MODELS:
        if key in tool_hint.lower():
            return FRONTEND_MODELS[key]
    return GENERIC_MODELS


def detect_frontend_from_flag(flag: str) -> list[str]:
    """Get model list from explicit --frontend flag."""
    flag = flag.lower().strip()
    for key, models in FRONTEND_MODELS.items():
        if key in flag or flag in key:
            return models
    return GENERIC_MODELS