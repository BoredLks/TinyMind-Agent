"""Token usage estimation.

The gateway's usage support over streaming is unknown, so we estimate from
character counts rather than risk altering the (working) request shape. The
estimate is rough and labelled as such in the UI; accurate counts can later
come from the provider's usage payload when available.
"""

from __future__ import annotations

from typing import List

_CHARS_PER_TOKEN = 2.5


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, round(len(text) / _CHARS_PER_TOKEN))


def estimate_tokens_for_messages(messages: List[dict]) -> int:
    return sum(estimate_tokens(m.get("content") or "") for m in messages)
