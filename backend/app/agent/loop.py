"""Agent tool-calling loop (production-grade, native tool-calling).

Drives multi-step tool use:
  model turn -> (text streamed via emit) -> tool_calls?
    no  -> final answer, return text
    yes -> for each call: emit tool_call, gate side-effects behind approval,
           run the tool, emit tool_result, feed the result back -> loop again
Bounded by max_iterations. Single tool failures degrade gracefully (the error
is fed back so the model can recover) rather than crashing the turn.

The adapter must expose `stream_turn(messages, tools)` yielding ("text", str)
and finally ("tool_calls", [ToolCallRequest, ...]) when tools are requested.
"""

from __future__ import annotations

import json
from typing import Awaitable, Callable, List, Optional

from app.agent.model.base import ToolCallRequest
from app.tools.base import ToolContext
from app.tools.registry import ToolRegistry

EmitFn = Callable[[dict], Awaitable[None]]
ApproveFn = Callable[[ToolCallRequest], Awaitable[bool]]


_DEFAULT_ARG_LEN = 4000
_DEFAULT_RESULT_LEN = 6000
_DEFAULT_TEXT_LEN = 8000


def _parse_args(raw: str) -> dict:
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _truncate(raw: str, limit: int) -> str:
    """Truncate a string for storage in message history to avoid context blowup.

    For JSON strings, truncate the inner content value while keeping valid JSON.
    """
    if not raw or len(raw) <= limit:
        return raw
    # Try to parse as JSON and truncate the largest content field
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            # Find the largest string field and truncate it
            largest_key = max(parsed.keys(), key=lambda k: len(str(parsed.get(k, ""))))
            largest_val = str(parsed[largest_key])
            if len(largest_val) > limit // 2:
                parsed[largest_key] = largest_val[:limit // 2] + f"... [truncated, {len(largest_val)} chars total]"
                result = json.dumps(parsed, ensure_ascii=False)
                if len(result) <= limit + 200:  # small overhead OK
                    return result
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return raw[:limit] + f"... [truncated, {len(raw)} chars total]"


async def run_agent_loop(
    adapter,
    registry: ToolRegistry,
    ctx: ToolContext,
    messages: List[dict],
    *,
    emit: EmitFn,
    request_approval: ApproveFn,
    disabled_tools: Optional[List[str]] = None,
    max_iterations: int = 30,
    max_tool_arg_len: int = _DEFAULT_ARG_LEN,
    max_tool_result_len: int = _DEFAULT_RESULT_LEN,
    max_tool_text_len: int = _DEFAULT_TEXT_LEN,
) -> str:
    """Run the loop. `messages` (OpenAI format) is mutated. Returns final text."""
    tools_schema = registry.openai_schema(disabled_tools) or None
    disabled_set = set(disabled_tools or [])
    last_text = ""

    for _ in range(max_iterations):
        text_parts: List[str] = []
        tool_calls: Optional[List[ToolCallRequest]] = None

        async for kind, payload in adapter.stream_turn(messages, tools_schema):
            if kind == "text":
                text_parts.append(payload)
                await emit({"type": "token", "delta": payload})
            elif kind == "reasoning":
                await emit({"type": "reasoning", "delta": payload})
            elif kind == "tool_calls":
                tool_calls = payload

        last_text = "".join(text_parts)
        if not tool_calls:
            return last_text

        messages.append(
            {
                "role": "assistant",
                "content": _truncate(last_text, max_tool_text_len) or None,
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {
                            "name": c.name,
                            "arguments": _truncate(c.arguments, max_tool_arg_len),
                        },
                    }
                    for c in tool_calls
                ],
            }
        )

        for call in tool_calls:
            await emit(
                {
                    "type": "tool_call",
                    "call_id": call.id,
                    "name": call.name,
                    "args": call.arguments,
                    "state": "running",
                }
            )

            if call.name in disabled_set:
                await _result(emit, messages, call, False, f"tool disabled: {call.name}", max_tool_result_len=max_tool_result_len)
                continue

            tool = registry.get(call.name)
            if tool is None:
                await _result(emit, messages, call, False, f"unknown tool: {call.name}", max_tool_result_len=max_tool_result_len)
                continue

            if tool.spec.requires_approval:
                approved = await request_approval(call)
                if not approved:
                    await _result(emit, messages, call, False, "用户拒绝执行该工具", max_tool_result_len=max_tool_result_len)
                    continue

            previous_call_id = ctx.current_call_id
            ctx.current_call_id = call.id
            try:
                result = await tool.run(_parse_args(call.arguments), ctx)
            except Exception as tool_exc:  # noqa: BLE001 — catch tool runtime errors
                ctx.current_call_id = previous_call_id
                await _result(emit, messages, call, False, f"工具执行异常: {tool_exc}", max_tool_result_len=max_tool_result_len)
                continue
            finally:
                ctx.current_call_id = previous_call_id
            content = result.content if result.ok else (result.error or "tool failed")
            await _result(emit, messages, call, result.ok, content, result.display, max_tool_result_len=max_tool_result_len)

    return last_text or "(已达到工具调用迭代上限)"


async def _result(
    emit: EmitFn,
    messages: List[dict],
    call: ToolCallRequest,
    ok: bool,
    content: str,
    display: Optional[dict] = None,
    max_tool_result_len: int = _DEFAULT_RESULT_LEN,
) -> None:
    event = {"type": "tool_result", "call_id": call.id, "name": call.name, "ok": ok, "content": content}
    if display is not None:
        event["display"] = display
    await emit(event)
    messages.append({"role": "tool", "tool_call_id": call.id, "content": _truncate(content, max_tool_result_len)})
