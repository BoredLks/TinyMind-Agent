"""Model adapter contract.

Every provider implementation (OpenAI-compatible now; others later) exposes the
same `stream_chat` async generator so the orchestration / API layers never care
which provider is behind it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class ToolCallRequest:
    """A tool call the model wants to make (assembled from a streamed response)."""

    id: str
    name: str
    arguments: str  # raw JSON string as emitted by the model


class ModelAdapter(ABC):
    """Streaming chat contract. Implementations yield content deltas."""

    @abstractmethod
    def stream_chat(
        self,
        messages: List[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Yield assistant text deltas as they arrive."""
        raise NotImplementedError
