"""
Robust context compactor.
Trims conversation history before sending to provider to prevent context overflow.

Strategy (in order of aggressiveness):
  1. Trim large tool result content older than KEEP_RECENT_TURNS
  2. Drop middle messages if still over limit, keeping first + last N
  3. Never touch: last KEEP_RECENT_TURNS, first message, tool call/result pairs
"""

import json
import logging

logger = logging.getLogger("prism.compactor")

# Config
KEEP_RECENT_TURNS = 10  # always preserve last N turn pairs
TOOL_RESULT_MAX_CHARS = 500  # truncate tool results older than KEEP_RECENT_TURNS to this
SOFT_TOKEN_LIMIT = 60_000  # start trimming above this (estimated)
HARD_TOKEN_LIMIT = 90_000  # drop middle messages above this


def estimate_tokens(obj) -> int:
    """Rough token estimate: 1 token ≈ 4 chars of JSON."""
    return len(json.dumps(obj, ensure_ascii=False)) // 4


def compact(messages: list, tools: list | None = None) -> tuple[list, bool]:
    """
    Compact a message list to fit within context limits.

    Returns:
        (compacted_messages, was_compacted)
    """
    if not messages:
        return messages, False

    total_est = estimate_tokens(messages) + estimate_tokens(tools or [])

    if total_est < SOFT_TOKEN_LIMIT:
        return messages, False  # nothing to do

    logger.info(f"Compacting: ~{total_est:,} estimated tokens, {len(messages)} messages")

    # Phase 1: trim large tool result content in older messages
    compacted = _trim_old_tool_results(messages)
    new_est = estimate_tokens(compacted) + estimate_tokens(tools or [])

    if new_est < SOFT_TOKEN_LIMIT:
        saved = total_est - new_est
        logger.info(f"Phase 1 done: ~{new_est:,} tokens (saved ~{saved:,})")
        return compacted, True

    # Phase 2: drop middle messages if still over limit
    if new_est >= HARD_TOKEN_LIMIT:
        compacted = _drop_middle(compacted)
        new_est = estimate_tokens(compacted) + estimate_tokens(tools or [])
        logger.info(f"Phase 2 done: ~{new_est:,} tokens, {len(compacted)} messages")
        return compacted, True

    return compacted, True


def _trim_old_tool_results(messages: list) -> list:
    """
    Truncate tool result content for messages older than KEEP_RECENT_TURNS.
    Keeps the tool_use_id and a summary so the model knows what happened.
    """
    if len(messages) <= KEEP_RECENT_TURNS * 2:
        return messages

    cutoff = max(0, len(messages) - (KEEP_RECENT_TURNS * 2))
    result = []

    for i, m in enumerate(messages):
        if i >= cutoff:
            result.append(m)
            continue

        content = m.get("content", "")

        if isinstance(content, list):
            new_blocks = []
            trimmed_any = False
            for block in content:
                if not isinstance(block, dict):
                    new_blocks.append(block)
                    continue

                if block.get("type") == "tool_result":
                    rc = block.get("content", "")
                    if isinstance(rc, str) and len(rc) > TOOL_RESULT_MAX_CHARS:
                        new_blocks.append({
                            **block,
                            "content": rc[:TOOL_RESULT_MAX_CHARS] + f"\n[...{len(rc) - TOOL_RESULT_MAX_CHARS} chars trimmed by prism]",
                        })
                        trimmed_any = True
                    elif isinstance(rc, list):
                        flat = "\n".join(
                            b.get("text", "") for b in rc
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                        if len(flat) > TOOL_RESULT_MAX_CHARS:
                            new_blocks.append({
                                **block,
                                "content": flat[:TOOL_RESULT_MAX_CHARS] + f"\n[...trimmed]",
                            })
                            trimmed_any = True
                        else:
                            new_blocks.append(block)
                    else:
                        new_blocks.append(block)
                else:
                    new_blocks.append(block)

            if trimmed_any:
                result.append({**m, "content": new_blocks})
            else:
                result.append(m)
        else:
            result.append(m)

    return result


def _drop_middle(messages: list) -> list:
    """
    Drop middle messages when still over hard limit.
    Always keeps: first message + last KEEP_RECENT_TURNS*2 messages.
    Inserts a placeholder so the model knows history was trimmed.
    
    CRITICAL: Preserves complete tool call/result chains.
    """
    keep_tail = KEEP_RECENT_TURNS * 2
    keep_head = 1

    if len(messages) <= keep_head + keep_tail:
        return messages

    head = messages[:keep_head]
    tail = messages[-keep_tail:]
    dropped = len(messages) - keep_head - keep_tail

    # Ensure tail doesn't start with an orphaned tool_result
    while tail and _is_orphaned_tool_result(tail[0], messages):
        tail = tail[1:]
        dropped += 1

    # Also check: last message in head shouldn't be a tool_use without result
    # If so, move more messages to preserve the pair
    while head and tail and _is_orphaned_tool_use(head[-1], tail[0]):
        # Move first of tail to head to complete the pair
        head.append(tail[0])
        tail = tail[1:]
        dropped -= 1

    placeholder = {
        "role": "user",
        "content": f"[Prism: {dropped} earlier messages were removed to fit context window. Continuing from recent context below.]",
    }

    logger.info(f"Dropped {dropped} middle messages")
    return head + [placeholder] + tail


def _is_orphaned_tool_result(msg: dict, all_messages: list) -> bool:
    """Check if a message is a tool_result with no preceding tool_use in the kept context."""
    content = msg.get("content", "")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tool_use_id = block.get("tool_use_id")
                # Check if any previous message has a matching tool_use
                if tool_use_id and not _has_matching_tool_use(tool_use_id, all_messages):
                    return True
    return msg.get("role") == "tool" and not _has_matching_tool_use(None, all_messages)


def _has_matching_tool_use(tool_use_id: str | None, messages: list) -> bool:
    """Check if any message in the list contains a matching tool_use."""
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    if tool_use_id is None or block.get("id") == tool_use_id:
                        return True
    return False


def _is_orphaned_tool_use(last_head_msg: dict, first_tail_msg: dict) -> bool:
    """
    Check if the last message in head is a tool_use that expects a tool_result
    that may have been dropped.
    """
    content = last_head_msg.get("content", "")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_id = block.get("id")
                # Check if first tail is the matching result
                tail_content = first_tail_msg.get("content", "")
                if isinstance(tail_content, list):
                    for tail_block in tail_content:
                        if (isinstance(tail_block, dict)
                                and tail_block.get("type") == "tool_result"
                                and tail_block.get("tool_use_id") == tool_id):
                            return False  # Has matching result
                return True  # No matching result found
    return False
