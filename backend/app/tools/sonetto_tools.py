"""SonettoHere-inspired compatible tools.

These are native SuperAgent tools that preserve the useful SonettoHere tool
surface where the behavior can be implemented without that project's runtime.
Third-party integrations that require unrelated SDK/API keys stay documented as
skills/tool docs instead of being registered as broken call targets.
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from typing import Any, Dict

from app.storage import dao

from .base import Tool, ToolContext, ToolResult, ToolSpec


def _ok(data: dict) -> str:
    return json.dumps({"success": True, "data": data}, ensure_ascii=False)


def _err(message: str) -> str:
    return json.dumps({"success": False, "error": message}, ensure_ascii=False)


def _db(ctx: ToolContext):
    conn = getattr(ctx, "db", None)
    if conn is None:
        raise RuntimeError("数据库上下文不可用")
    return conn


class TimeTool(Tool):
    spec = ToolSpec(
        name="time_tool",
        description="获取当前日期和时间。迁移自 SonettoHere system/time_tool。",
        parameters={"type": "object", "properties": {}},
        doc="# time_tool\n\n获取当前日期、时间、星期和时区。无参数。",
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        now = datetime.now()
        content = _ok(
            {
                "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "weekday": now.strftime("%A"),
                "timezone": "Asia/Shanghai",
            }
        )
        return ToolResult(ok=True, content=content, display={"kind": "json", "data": json.loads(content)})


_ANSWERS = [
    "是的，但先把条件看清楚。",
    "现在还不是最佳时机。",
    "答案已经在你的第一个直觉里。",
    "可以尝试，但要留一个回退方案。",
    "先等待一个更明确的信号。",
    "不要把临时情绪当作长期结论。",
    "值得做，只是步骤要拆小。",
    "换一个角度，答案会变得简单。",
]


class AnswerBookTool(Tool):
    spec = ToolSpec(
        name="answer_book",
        description="答案之书：提出问题，获得随机启发式回答。迁移自 SonettoHere entertainment/answer_book。",
        parameters={
            "type": "object",
            "properties": {"question": {"type": "string", "description": "想询问的问题"}},
            "required": ["question"],
        },
        doc="# answer_book\n\n娱乐型工具。输入一个问题，返回一句随机启发式答案；不依赖外部 API。",
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        question = str(args.get("question") or "").strip()
        if not question:
            return ToolResult(ok=False, content="", error=_err("问题不能为空"))
        answer = random.choice(_ANSWERS)
        content = _ok({"question": question, "answer": answer})
        return ToolResult(ok=True, content=content, display={"kind": "json", "data": json.loads(content)})


_MAJOR_ARCANA = [
    ("愚人", "新的开始、冒险、可能性"),
    ("魔术师", "创造力、行动、资源整合"),
    ("女祭司", "直觉、沉静、隐藏信息"),
    ("女皇", "丰盛、滋养、创造"),
    ("皇帝", "秩序、责任、结构"),
    ("教皇", "传统、学习、精神指导"),
    ("恋人", "关系、选择、价值观"),
    ("战车", "意志、推进、胜利"),
    ("力量", "勇气、耐心、温柔的控制"),
    ("隐士", "内省、独处、寻找真相"),
    ("命运之轮", "变化、周期、机会"),
    ("正义", "平衡、公正、因果"),
    ("倒吊人", "暂停、换位、等待"),
    ("死神", "结束、转变、释放"),
    ("节制", "调和、耐心、适度"),
    ("恶魔", "束缚、欲望、执念"),
    ("塔", "突变、破旧、觉醒"),
    ("星星", "希望、疗愈、信心"),
    ("月亮", "不确定、潜意识、迷雾"),
    ("太阳", "清晰、快乐、成功"),
    ("审判", "召唤、复盘、更新"),
    ("世界", "完成、整合、圆满"),
]


class TarotTool(Tool):
    spec = ToolSpec(
        name="tarot",
        description="简化韦特塔罗占卜，支持 single/three。迁移自 SonettoHere entertainment/tarot 的本地无依赖版本。",
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "spread_type": {"type": "string", "enum": ["single", "three"], "default": "three"},
            },
            "required": ["question"],
        },
        doc="# tarot\n\n娱乐型工具。`spread_type=single` 抽一张，`three` 抽过去/现在/未来三张。",
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        question = str(args.get("question") or "").strip()
        spread_type = str(args.get("spread_type") or "three")
        if not question:
            return ToolResult(ok=False, content="", error=_err("占卜问题不能为空"))
        count = 1 if spread_type == "single" else 3
        positions = ["当前问题"] if count == 1 else ["过去", "现在", "未来"]
        cards = []
        for position, card in zip(positions, random.sample(_MAJOR_ARCANA, count)):
            reversed_card = random.random() < 0.15
            cards.append(
                {
                    "position": position,
                    "name": card[0],
                    "status": "逆位" if reversed_card else "正位",
                    "keywords": card[1],
                }
            )
        content = _ok({"question": question, "spread_type": spread_type, "cards": cards})
        return ToolResult(ok=True, content=content, display={"kind": "json", "data": json.loads(content)})


class ListMemoriesTool(Tool):
    spec = ToolSpec(
        name="list_memories",
        description="列出 SuperAgent 长期记忆。兼容 SonettoHere memory/list_memories。",
        parameters={"type": "object", "properties": {}},
        doc="# list_memories\n\n列出当前 SQLite 长期记忆条目。",
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        memories = dao.list_memories(_db(ctx))
        content = _ok({"count": len(memories), "items": memories})
        return ToolResult(ok=True, content=content, display={"kind": "json", "data": json.loads(content)})


class ReadMemoriesTool(Tool):
    spec = ToolSpec(
        name="read_memories",
        description="按 ID 读取长期记忆完整内容。兼容 SonettoHere memory/read_memories。",
        parameters={
            "type": "object",
            "properties": {"ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["ids"],
        },
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        ids = args.get("ids") or []
        items = [dao.get_memory(_db(ctx), str(memory_id)) for memory_id in ids]
        found = [item for item in items if item is not None]
        content = _ok({"count": len(found), "items": found})
        return ToolResult(ok=True, content=content, display={"kind": "json", "data": json.loads(content)})


class CreateMemoryTool(Tool):
    spec = ToolSpec(
        name="create_memory",
        description="手动创建一条长期记忆。兼容 SonettoHere memory/create_memory。",
        parameters={
            "type": "object",
            "properties": {"theme": {"type": "string"}, "content": {"type": "string"}},
            "required": ["content"],
        },
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        content_text = str(args.get("content") or "").strip()
        if not content_text:
            return ToolResult(ok=False, content="", error=_err("content 不能为空"))
        memory = dao.add_memory(_db(ctx), str(args.get("theme") or "偏好").strip() or "偏好", content_text)
        content = _ok({"memory": memory})
        return ToolResult(ok=True, content=content, display={"kind": "json", "data": json.loads(content)})


class UpdateMemoryTool(Tool):
    spec = ToolSpec(
        name="update_memory",
        description="更新一条长期记忆。兼容 SonettoHere memory/update_memory。",
        parameters={
            "type": "object",
            "properties": {"id": {"type": "string"}, "theme": {"type": "string"}, "content": {"type": "string"}},
            "required": ["id"],
        },
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        memory = dao.update_memory(
            _db(ctx),
            str(args.get("id") or ""),
            theme=str(args["theme"]).strip() if args.get("theme") is not None else None,
            content=str(args["content"]).strip() if args.get("content") is not None else None,
        )
        if memory is None:
            return ToolResult(ok=False, content="", error=_err("memory not found"))
        content = _ok({"memory": memory})
        return ToolResult(ok=True, content=content, display={"kind": "json", "data": json.loads(content)})


class DeleteMemoryTool(Tool):
    spec = ToolSpec(
        name="delete_memory",
        description="删除一条长期记忆。兼容 SonettoHere memory/delete_memory。",
        parameters={"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        ok = dao.delete_memory(_db(ctx), str(args.get("id") or ""))
        if not ok:
            return ToolResult(ok=False, content="", error=_err("memory not found"))
        return ToolResult(ok=True, content=_ok({"deleted": True}))


class MergeMemoriesTool(Tool):
    spec = ToolSpec(
        name="merge_memories",
        description="合并两条长期记忆。兼容 SonettoHere memory/merge_memories。",
        parameters={
            "type": "object",
            "properties": {
                "keep_id": {"type": "string"},
                "remove_id": {"type": "string"},
                "theme": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["keep_id", "remove_id", "content"],
        },
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        ok = dao.merge_memories(
            _db(ctx),
            str(args.get("keep_id") or ""),
            str(args.get("remove_id") or ""),
            theme=str(args.get("theme") or "偏好").strip() or "偏好",
            content=str(args.get("content") or "").strip(),
        )
        if not ok:
            return ToolResult(ok=False, content="", error=_err("merge failed"))
        return ToolResult(ok=True, content=_ok({"merged": True}))


def sonetto_compatible_tools() -> list[Tool]:
    return [
        TimeTool(),
        AnswerBookTool(),
        TarotTool(),
        ListMemoriesTool(),
        ReadMemoriesTool(),
        CreateMemoryTool(),
        UpdateMemoryTool(),
        DeleteMemoryTool(),
        MergeMemoriesTool(),
    ]
