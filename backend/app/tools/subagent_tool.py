"""dispatch_subagent tool — delegate an independent subtask to an isolated agent.

Requires approval (the user sees the subtask before it runs). The actual
isolated run is performed by ctx.subagent_runner (set per turn by the chat
endpoint), which also blocks recursion.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import Tool, ToolContext, ToolResult, ToolSpec


class DispatchSubagentTool(Tool):
    spec = ToolSpec(
        name="dispatch_subagent",
        description=(
            "把一个独立子任务派发给上下文隔离的子代理执行并返回结果。"
            "用于可独立完成的子任务；或派发审查子代理(role=reviewer)做两阶段审查。"
            "task 必须自包含——子代理看不到当前对话。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "自包含的完整子任务描述"},
                "role": {
                    "type": "string",
                    "enum": ["implementer", "reviewer"],
                    "description": "子代理角色，默认 implementer",
                },
            },
            "required": ["task"],
        },
        requires_approval=True,
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        runner = getattr(ctx, "subagent_runner", None)
        if runner is None:
            return ToolResult(ok=False, content="", error="子代理系统不可用")
        task = str(args.get("task", "")).strip()
        if not task:
            return ToolResult(ok=False, content="", error="缺少 task 参数")
        role = str(args.get("role", "implementer"))
        try:
            result = await runner.run(task, role)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(ok=False, content="", error=f"子代理执行失败：{exc}")
        child_session = getattr(runner, "last_session", None)
        display = {"kind": "subagent", "role": role, "task": task, "result": result or ""}
        if child_session is not None:
            display["session_id"] = child_session.get("id")
            display["title"] = child_session.get("title")
        return ToolResult(ok=True, content=result or "(子代理无输出)", display=display)
