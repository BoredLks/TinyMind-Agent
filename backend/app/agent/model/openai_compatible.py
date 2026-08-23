"""OpenAI-compatible streaming adapter.

Works against any endpoint that speaks the OpenAI Chat Completions API:
OpenAI itself, local servers (Ollama / LM Studio), or a custom gateway
(the default in M1). Only the base_url / api_key / model differ.
"""

from __future__ import annotations

from typing import AsyncIterator, List, Optional

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from app.agent.retry import with_retry
from .base import ChatMessage, ModelAdapter, ToolCallRequest

# Transient errors worth retrying with backoff (not 4xx like auth/bad-request).
_RETRYABLE = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)


class OpenAICompatibleAdapter(ModelAdapter):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 300.0,
    ) -> None:
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._on_retry = None  # async callable(attempt:int), set per turn

    def set_retry_callback(self, callback) -> None:
        self._on_retry = callback

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        params = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "temperature": self._temperature if temperature is None else temperature,
        }
        effective_max = self._max_tokens if max_tokens is None else max_tokens
        if effective_max is not None:
            params["max_tokens"] = effective_max

        stream = await self._client.chat.completions.create(**params)
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    async def stream_turn(self, messages, tools=None):
        """One model turn over OpenAI-format message dicts, with optional tools.

        Yields ("text", delta) for content and, at the end, ("tool_calls", [...])
        if the model requested tool calls (arguments assembled across chunks).
        """
        params = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "temperature": self._temperature,
        }
        if self._max_tokens is not None:
            params["max_tokens"] = self._max_tokens
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        acc: dict = {}
        import asyncio as _asyncio

        # Per-chunk timeout: if no data arrives for this long, consider stream dead.
        # Prevents indefinite hangs when the gateway silently drops connections.
        _CHUNK_TIMEOUT = 120  # seconds

        async def _create():
            return await self._client.chat.completions.create(**params)

        stream = await with_retry(_create, retryable=_RETRYABLE, on_retry=self._on_retry)
        got_output = False
        _aiter = stream.__aiter__()
        try:
            while True:
                try:
                    chunk = await _asyncio.wait_for(_aiter.__anext__(), timeout=_CHUNK_TIMEOUT)
                except StopAsyncIteration:
                    break
                except _asyncio.TimeoutError:
                    raise RuntimeError(
                        f"模型 {self._model} 流式响应超时：{_CHUNK_TIMEOUT}秒内未收到数据。"
                        "可能是网关超时或模型响应过慢，请重试。"
                    )
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    got_output = True
                    yield ("text", delta.content)
                elif delta and getattr(delta, "reasoning_content", None):
                    got_output = True
                    yield ("reasoning", delta.reasoning_content)
                if delta and getattr(delta, "tool_calls", None):
                    got_output = True
                    for tcd in delta.tool_calls:
                        idx = tcd.index if tcd.index is not None else 0
                        slot = acc.setdefault(idx, {"id": None, "name": "", "args": ""})
                        if tcd.id:
                            slot["id"] = tcd.id
                        fn = getattr(tcd, "function", None)
                        if fn:
                            if fn.name:
                                slot["name"] += fn.name
                            if fn.arguments:
                                slot["args"] += fn.arguments
        except _asyncio.TimeoutError:
            raise RuntimeError(
                f"模型 {self._model} 流式响应超时：{_CHUNK_TIMEOUT}秒内未收到数据。"
                "可能是网关超时或模型响应过慢，请重试。"
            )

        if acc:
            # Fall back to a globally-unique id when the model omits one, so
            # approval/interaction routing never collides across concurrent turns.
            import uuid as _uuid

            calls = [
                ToolCallRequest(
                    id=slot["id"] or f"call_{_uuid.uuid4().hex[:8]}_{i}",
                    name=slot["name"],
                    arguments=slot["args"],
                )
                for i, slot in sorted(acc.items())
            ]
            yield ("tool_calls", calls)
        elif not got_output:
            raise RuntimeError(
                f"模型 {self._model} 仅返回推理内容，无实际输出。"
                "请重试或切换模型。"
            )
