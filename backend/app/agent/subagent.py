"""Subagent dispatch: run an isolated agent loop for an independent subtask.

The controller delegates a self-contained task to a fresh agent (its own
message history, a focused worker persona, a new adapter instance) sharing the
same tools/workspace. Side-effecting tools still flow through the controller's
approval gate. Recursion is blocked (a subagent cannot dispatch subagents).
"""

from __future__ import annotations

import sqlite3

from app.agent.loop import run_agent_loop
from app.storage import dao

_PERSONAS = {
    "implementer": (
        "你是一个专注的实现子代理，上下文与主对话隔离。只完成被指派的具体子任务："
        "理解需求 → 实现 →（如涉及代码）测试与验证 → 简洁汇报你做了什么、结果如何、有无遗留。"
        "搞不定就如实说明，不要假装完成。"
    ),
    "reviewer": (
        "你是一个独立的审查子代理。核查实现是否符合要求：有无遗漏、有无多余、质量如何。"
        "不要信任他人的报告——自己读代码核实。用 file:line 指出具体问题，并给出明确结论。"
    ),
}

_MAX_SUBAGENT_DEPTH = 2


class SubagentRunner:
    def __init__(
        self,
        adapter_factory,
        registry,
        ctx,
        emit,
        request_approval,
        base_disabled=None,
        *,
        conn: sqlite3.Connection | None = None,
    ):
        self._adapter_factory = adapter_factory
        self._registry = registry
        self._ctx = ctx
        self._emit = emit
        self._request_approval = request_approval
        self._base_disabled = list(base_disabled or [])
        self._conn = conn
        self.last_session: dict | None = None

    async def run(self, task: str, role: str = "implementer") -> str:
        current_depth = int(getattr(self._ctx, "subagent_depth", 0) or 0)
        if current_depth >= _MAX_SUBAGENT_DEPTH:
            raise RuntimeError(f"子代理嵌套深度已达到上限：{_MAX_SUBAGENT_DEPTH}")

        persona = _PERSONAS.get(role, _PERSONAS["implementer"])
        messages = [
            {"role": "system", "content": persona},
            {"role": "user", "content": task},
        ]
        depth = current_depth + 1
        parent_session_id = getattr(self._ctx, "session_id", None)
        child_session = self._create_child_session(task, role, persona, parent_session_id, depth)
        self.last_session = child_session
        if child_session is not None:
            await self._emit(
                {
                    "type": "sub_session_created",
                    "session_id": child_session["id"],
                    "parent_session_id": parent_session_id,
                    "title": child_session["title"],
                    "role": role,
                    "task": task,
                    "depth": depth,
                }
            )

        async def sub_emit(event: dict) -> None:
            # Surface the subagent's tool activity (tagged), not its token stream.
            if event.get("type") in (
                "tool_call",
                "tool_result",
                "tool_approval_request",
                "interaction_request",
            ):
                await self._emit({**event, "subagent": True})

        adapter = self._adapter_factory()  # fresh adapter = isolated context
        disabled = list(self._base_disabled)
        if depth >= _MAX_SUBAGENT_DEPTH and "dispatch_subagent" not in disabled:
            disabled.append("dispatch_subagent")

        previous_depth = int(getattr(self._ctx, "subagent_depth", 0) or 0)
        previous_session_id = getattr(self._ctx, "session_id", None)
        self._ctx.subagent_depth = depth
        if child_session is not None:
            self._ctx.session_id = child_session["id"]
        try:
            result = await run_agent_loop(
                adapter,
                self._registry,
                self._ctx,
                messages,
                emit=sub_emit,
                request_approval=self._request_approval,
                disabled_tools=disabled,
            )
        except Exception as exc:
            self._finish_child_session(child_session, ok=False, result=f"子代理执行失败：{exc}")
            await self._emit(
                {
                    "type": "sub_session_done",
                    "session_id": child_session["id"] if child_session else None,
                    "parent_session_id": parent_session_id,
                    "role": role,
                    "ok": False,
                    "error": str(exc),
                    "depth": depth,
                }
            )
            raise
        finally:
            self._ctx.subagent_depth = previous_depth
            self._ctx.session_id = previous_session_id

        self._finish_child_session(child_session, ok=True, result=result)
        await self._emit(
            {
                "type": "sub_session_done",
                "session_id": child_session["id"] if child_session else None,
                "parent_session_id": parent_session_id,
                "role": role,
                "ok": True,
                "depth": depth,
            }
        )
        return result

    def _create_child_session(
        self,
        task: str,
        role: str,
        persona: str,
        parent_session_id: str | None,
        depth: int,
    ) -> dict | None:
        if self._conn is None or not parent_session_id:
            return None
        title = _child_title(role, task)
        parent = dao.get_session(self._conn, parent_session_id)
        project_id = parent.get("project_id") if parent else None
        session = dao.create_session(
            self._conn,
            title=title,
            system_prompt=persona,
            meta_json={
                "is_subagent": True,
                "parent_session_id": parent_session_id,
                "role": role,
                "task": task,
                "status": "running",
                "depth": depth,
            },
            project_id=project_id,
        )
        dao.add_message(self._conn, session["id"], "user", task)
        return session

    def _finish_child_session(self, child_session: dict | None, *, ok: bool, result: str) -> None:
        if self._conn is None or child_session is None:
            return
        dao.add_message(self._conn, child_session["id"], "assistant", result or "")
        meta = {
            "is_subagent": True,
            "parent_session_id": _meta_parent(child_session),
            "role": _meta_value(child_session, "role"),
            "task": _meta_value(child_session, "task"),
            "status": "done" if ok else "failed",
            "depth": _meta_value(child_session, "depth"),
        }
        dao.set_session_meta(self._conn, child_session["id"], meta)


def _child_title(role: str, task: str) -> str:
    label = "审查" if role == "reviewer" else "执行"
    one_line = " ".join(task.split())
    suffix = one_line[:32] + ("…" if len(one_line) > 32 else "")
    return f"子代理/{label} · {suffix or '未命名任务'}"


def _meta_value(session: dict, key: str):
    import json

    try:
        meta = json.loads(session.get("meta_json") or "{}")
    except json.JSONDecodeError:
        return None
    return meta.get(key)


def _meta_parent(session: dict) -> str | None:
    parent = _meta_value(session, "parent_session_id")
    return str(parent) if parent else None
