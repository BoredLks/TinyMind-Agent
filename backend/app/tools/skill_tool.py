"""load_skill tool — progressive disclosure of skill bodies.

Read-only (no approval). The agent calls this when a skill in the system-prompt
index looks relevant; it returns the skill's full Markdown body to follow.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import Tool, ToolContext, ToolResult, ToolSpec


class LoadSkillTool(Tool):
    spec = ToolSpec(
        name="load_skill",
        description=(
            "加载一个技能的完整内容（方法论指南）并据此行动。"
            "当某个列出的技能可能适用于当前任务时调用它。"
        ),
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string", "description": "技能名称"}},
            "required": ["name"],
        },
        requires_approval=False,
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        skills = getattr(ctx, "skills", None)
        name = str(args.get("name", "")).strip()
        if skills is None:
            return ToolResult(ok=False, content="", error="技能系统不可用")
        body = skills.get_body(name)
        if body is None:
            return ToolResult(ok=False, content="", error=f"未找到技能：{name}")
        return ToolResult(ok=True, content=body)
