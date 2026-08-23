"""Interactive tools that pause the agent and wait for user input."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .base import Tool, ToolContext, ToolResult, ToolSpec


def _json_result(data: dict) -> str:
    return json.dumps({"success": True, "data": data}, ensure_ascii=False)


class _AskUserBase(Tool):
    mode = "qa"

    async def _ask(
        self,
        ctx: ToolContext,
        *,
        question: str,
        options: List[str] | None = None,
    ) -> ToolResult:
        if ctx.interact is None:
            return ToolResult(False, "", "当前通道不支持等待用户输入")
        question = question.strip()
        if not question:
            return ToolResult(False, "", "question 不能为空")
        options = options or []
        if self.mode in {"single_choice", "multi_choice"} and not options:
            return ToolResult(False, "", "options 不能为空")

        response = await ctx.interact(
            {
                "tool_name": self.spec.name,
                "mode": self.mode,
                "question": question,
                "options": options,
            }
        )
        return ToolResult(
            True,
            _json_result({"question": question, "answer": response, "options": options}),
            display={
                "kind": "interaction_answer",
                "mode": self.mode,
                "question": question,
                "answer": response,
            },
        )


class AskUserQATool(_AskUserBase):
    mode = "qa"
    spec = ToolSpec(
        name="ask_user_qa",
        description="暂停当前 Agent，向用户提出一个自由文本问题并等待回复。",
        parameters={
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
        doc=(
            "# ask_user_qa\n\n"
            "当缺少用户偏好、账号选择、路径意图、需求取舍等关键信息时调用。"
            "问题必须具体，一次只问必要信息。"
        ),
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        return await self._ask(ctx, question=str(args.get("question", "")))


class AskUserSingleChoiceTool(_AskUserBase):
    mode = "single_choice"
    spec = ToolSpec(
        name="ask_user_single_choice",
        description="暂停当前 Agent，让用户从多个选项中选择一个。",
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question", "options"],
        },
        doc=(
            "# ask_user_single_choice\n\n"
            "用于需要用户做唯一选择的场景。选项应短而互斥，并把推荐项放在第一项。"
        ),
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        options = args.get("options")
        return await self._ask(
            ctx,
            question=str(args.get("question", "")),
            options=[str(x) for x in options] if isinstance(options, list) else [],
        )


class AskUserMultiChoiceTool(_AskUserBase):
    mode = "multi_choice"
    spec = ToolSpec(
        name="ask_user_multi_choice",
        description="暂停当前 Agent，让用户从多个选项中选择一个或多个。",
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question", "options"],
        },
        doc=(
            "# ask_user_multi_choice\n\n"
            "用于用户可以同时保留多个偏好的场景。选项要少而清楚，避免一次列出过长列表。"
        ),
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        options = args.get("options")
        return await self._ask(
            ctx,
            question=str(args.get("question", "")),
            options=[str(x) for x in options] if isinstance(options, list) else [],
        )
