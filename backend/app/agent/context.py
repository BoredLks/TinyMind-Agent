"""Context window management.

Supports two strategies:
- sliding_window: keep the most recent `max_messages` turns
- token_limit: keep as many recent messages as fit within `max_tokens`
"""

from __future__ import annotations

from typing import List, Optional

from app.agent.model.base import ChatMessage


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~0.75 tokens per char for mixed CJK/English."""
    return max(1, int(len(text) * 0.75))


def build_context(
    history: List[ChatMessage],
    system_prompt: Optional[str] = None,
    max_messages: int = 20,
    strategy: str = "sliding_window",
    max_tokens: int = 0,
) -> List[ChatMessage]:
    """Build the message list to send to the model."""
    convo = list(history)

    if strategy == "token_limit" and max_tokens > 0:
        kept: List[ChatMessage] = []
        token_budget = max_tokens
        for msg in reversed(convo):
            msg_tokens = _estimate_tokens(msg.content)
            if msg_tokens > token_budget:
                break
            kept.append(msg)
            token_budget -= msg_tokens
        kept.reverse()
        convo = kept
    elif strategy == "sliding_window" and max_messages > 0 and len(convo) > max_messages:
        convo = convo[-max_messages:]

    messages: List[ChatMessage] = []
    if system_prompt:
        messages.append(ChatMessage("system", system_prompt))
    messages.extend(convo)
    return messages