"""Tool documentation loader.

Mirrors SonettoHere's two-step tool pattern: tools stay short in the system
prompt, and the model can fetch richer tool/domain guidance only when needed.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import Tool, ToolContext, ToolResult, ToolSpec


class LoadToolDocTool(Tool):
    spec = ToolSpec(
        name="load_tool_doc",
        description=(
            "按需加载某个工具的完整说明、限制和示例。首次使用复杂工具、"
            "工具说明要求先读文档，或调用失败后需要排查时使用。"
        ),
        parameters={
            "type": "object",
            "properties": {"tool_name": {"type": "string", "description": "要读取说明的工具名"}},
            "required": ["tool_name"],
        },
        requires_approval=False,
        doc=(
            "# load_tool_doc\n\n"
            "读取工具的详细说明。不要把它当成业务工具；它只用于在调用其他工具前补充上下文。"
        ),
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        registry = getattr(ctx, "registry", None)
        if registry is None:
            return ToolResult(False, "", "工具注册表不可用")
        name = str(args.get("tool_name", "")).strip()
        if not name:
            return ToolResult(False, "", "tool_name 不能为空")
        doc = registry.doc_for(name)
        if doc is None:
            return ToolResult(False, "", f"未找到工具：{name}")
        return ToolResult(
            True,
            doc,
            display={"kind": "tool_doc", "tool_name": name, "chars": len(doc)},
        )
