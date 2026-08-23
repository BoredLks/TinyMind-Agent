"""SonettoHere task tracker — migrated to SuperAgent plugin.

- task_tracker: 无状态任务清单追踪工具
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from app.tools.base import Tool, ToolContext, ToolResult, ToolSpec


def _ok(data: dict) -> str:
    return json.dumps({"success": True, "data": data}, ensure_ascii=False)


def _err(message: str) -> str:
    return json.dumps({"success": False, "error": message}, ensure_ascii=False)


class TaskTrackerTool(Tool):
    spec = ToolSpec(
        name="task_tracker",
        description=(
            "无状态任务清单追踪。每次传入完整的 todos 列表（JSON 数组），"
            "工具返回统计摘要：总数、各状态计数、当前任务等。"
            "适合在多步骤任务中保持进度跟踪。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "全量任务清单",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "任务描述"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                                "description": "任务状态",
                            },
                            "activeForm": {"type": "string", "description": "进行中的动名词描述（可选）"},
                        },
                        "required": ["content", "status"],
                    },
                },
            },
            "required": ["todos"],
        },
        doc=(
            "# task_tracker\n\n"
            "无状态任务清单追踪。LLM 每次调用传入全量 todos 列表，工具返回统计摘要。"
            "不维护内部状态。"
        ),
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        todos = args.get("todos")
        if not todos or not isinstance(todos, list):
            return ToolResult(False, "", "请传入 todos 参数（全量任务清单）")

        total = len(todos)
        pending = sum(1 for t in todos if t.get("status") == "pending")
        in_progress = sum(1 for t in todos if t.get("status") == "in_progress")
        completed = sum(1 for t in todos if t.get("status") == "completed")
        current_task = next((t.get("content") for t in todos if t.get("status") == "in_progress"), None)

        result = {
            "total": total,
            "pending": pending,
            "in_progress": in_progress,
            "completed": completed,
            "current_task": current_task,
            "todos": todos,
        }
        return ToolResult(True, _ok(result), display={"kind": "json", "data": result})


def register(registry):
    registry.register(TaskTrackerTool())