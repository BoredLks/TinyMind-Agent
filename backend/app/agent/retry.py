"""Async retry with exponential backoff for transient provider errors.

Generic so it's unit-testable without the network: callers pass the awaitable
factory, the retryable exception types, and an optional async on_retry hook
(used to surface a "retrying" event to the UI).
"""

from __future__ import annotations

from typing import Awaitable, Callable, Optional, Tuple, Type, TypeVar

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

T = TypeVar("T")


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    retryable: Tuple[Type[BaseException], ...],
    max_attempts: int = 2,
    wait_multiplier: float = 0.5,
    wait_max: float = 10.0,
    on_retry: Optional[Callable[[int], Awaitable[None]]] = None,
) -> T:
    """Call fn(), retrying on `retryable` errors with exponential backoff.

    on_retry(attempt_number) is awaited before each retry (attempts 2+).
    Reraises the last exception once attempts are exhausted.
    """
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=wait_multiplier, max=wait_max),
        retry=retry_if_exception_type(retryable),
        reraise=True,
    ):
        with attempt:
            number = attempt.retry_state.attempt_number
            if number > 1 and on_retry is not None:
                await on_retry(number)
            return await fn()
    raise RuntimeError("unreachable")  # AsyncRetrying always returns or raises
