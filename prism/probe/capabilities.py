"""
Model capabilities learner.
Detects thinking blocks, content formats, etc from actual responses.
No hardcoding — learns from what the model actually sends.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("prism.probe.capabilities")


@dataclass
class ModelCapabilities:
    model:        str
    thinks:       bool       = False
    think_field:  str        = "thinking"   # field name inside thinking block
    think_type:   str        = "thinking"   # type value of thinking block
    learned:      bool       = False

    def __repr__(self):
        return f"<Caps model={self.model} thinks={self.thinks} field={self.think_field}>"


# Session-level capability cache — survives across requests
_caps: dict[str, ModelCapabilities] = {}


def get_capabilities(model: str) -> ModelCapabilities:
    if model not in _caps:
        _caps[model] = ModelCapabilities(model=model)
    return _caps[model]


def learn_from_chunk(model: str, chunk: dict) -> ModelCapabilities:
    """
    Learn model capabilities from a streaming chunk or full response.
    Called on every chunk — idempotent once learned.
    """
    caps = get_capabilities(model)
    if caps.learned:
        return caps

    # Check choices[0].delta for streaming
    delta = (chunk.get("choices") or [{}])[0].get("delta", {})

    # Check for thinking in delta content array
    delta_content = delta.get("content") or []
    if isinstance(delta_content, list):
        for block in delta_content:
            if isinstance(block, dict) and block.get("type") in ("thinking", "reasoning"):
                caps.thinks     = True
                caps.think_type = block.get("type", "thinking")
                caps.think_field = "thinking" if "thinking" in block else "text"
                caps.learned    = True
                logger.info(f"Learned: {model} THINKS via content array (type={caps.think_type} field={caps.think_field})")
                return caps

    # Check for DeepSeek/QwQ style — reasoning_content at delta level
    if delta.get("reasoning_content") is not None:
        caps.thinks      = True
        caps.think_type  = "reasoning"
        caps.think_field = "reasoning_content"
        caps.learned     = True
        logger.info(f"Learned: {model} THINKS via reasoning_content")
        return caps

    # Check full response content array (non-streaming)
    content = chunk.get("content") or []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("thinking", "reasoning"):
                caps.thinks     = True
                caps.think_type = block.get("type", "thinking")
                caps.think_field = "thinking" if "thinking" in block else "text"
                caps.learned    = True
                logger.info(f"Learned: {model} THINKS via full content (type={caps.think_type})")
                return caps

    # Check choices[0].message for non-streaming
    msg = (chunk.get("choices") or [{}])[0].get("message", {})
    msg_content = msg.get("content") or []
    if isinstance(msg_content, list):
        for block in msg_content:
            if isinstance(block, dict) and block.get("type") in ("thinking", "reasoning"):
                caps.thinks     = True
                caps.think_type = block.get("type", "thinking")
                caps.think_field = "thinking" if "thinking" in block else "text"
                caps.learned    = True
                logger.info(f"Learned: {model} THINKS via message content")
                return caps

    return caps


def extract_thinking_text(block: dict, caps: ModelCapabilities) -> str | None:
    """Extract thinking text from a block using learned field name."""
    if block.get("type") not in ("thinking", "reasoning"):
        return None
    return block.get(caps.think_field) or block.get("thinking") or block.get("text") or None