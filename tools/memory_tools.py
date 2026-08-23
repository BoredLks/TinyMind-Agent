"""SonettoHere memory tools — migrated to SuperAgent plugin.

- merge_memories: 合并重复/相似的长期记忆条目
- read_memories: 按主题/关键词查询长期记忆
"""

from __future__ import annotations

import json
from typing import Any, Dict

from app.tools.base import Tool, ToolContext, ToolResult, ToolSpec


def _ok(data: dict) -> str:
    return json.dumps({"success": True, "data": data}, ensure_ascii=False)


class MergeMemoriesTool(Tool):
    spec = ToolSpec(
        name="merge_memories",
        description=(
            "合并重复/相似的长期记忆条目。"
            "指定要保留的记忆 ID 和要合并的记忆 ID 列表，合并后删除被合并条目。"
            "可选指定新的合并后内容。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "keep_id": {"type": "string", "description": "保留的记忆 ID"},
                "merge_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要合并（删除）的记忆 ID 列表",
                },
                "new_content": {"type": "string", "description": "合并后的新内容（可选，不填则保留 keep_id 的原内容）"},
            },
            "required": ["keep_id", "merge_ids"],
        },
        doc="# merge_memories\n\n合并重复/相似的长期记忆。保留 keep_id 条目，删除 merge_ids 中的条目。",
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        keep_id = str(args.get("keep_id") or "").strip()
        merge_ids = args.get("merge_ids") or []
        new_content = str(args.get("new_content") or "").strip()

        if not keep_id:
            return ToolResult(False, "", "keep_id 不能为空")
        if not merge_ids or not isinstance(merge_ids, list):
            return ToolResult(False, "", "merge_ids 必须是非空列表")

        conn = getattr(ctx, "db", None)
        if conn is None:
            return ToolResult(False, "", "数据库不可用")

        try:
            # Verify keep_id exists
            keep = conn.execute("SELECT * FROM memories WHERE id = ?", (keep_id,)).fetchone()
            if not keep:
                return ToolResult(False, "", f"未找到保留记忆: {keep_id}")

            # Update content if provided
            if new_content:
                from datetime import datetime
                conn.execute(
                    "UPDATE memories SET content = ?, updated_at = ? WHERE id = ?",
                    (new_content, datetime.now().isoformat(), keep_id),
                )

            # Delete merged entries
            deleted = 0
            for mid in merge_ids:
                mid_str = str(mid).strip()
                if mid_str == keep_id:
                    continue
                cur = conn.execute("DELETE FROM memories WHERE id = ?", (mid_str,))
                deleted += cur.rowcount

            conn.commit()
            result = {"kept_id": keep_id, "deleted_count": deleted, "new_content": new_content or "(unchanged)"}
            return ToolResult(True, _ok(result), display={"kind": "json", "data": result})
        except Exception as exc:
            return ToolResult(False, "", f"合并失败: {exc}")


class ReadMemoriesTool(Tool):
    spec = ToolSpec(
        name="read_memories",
        description=(
            "按主题或关键词查询长期记忆。"
            "返回匹配的记忆列表，支持模糊搜索。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词（模糊匹配 theme 和 content）"},
                "limit": {"type": "integer", "description": "返回数量限制，默认 20"},
            },
        },
        doc="# read_memories\n\n按主题/关键词查询长期记忆，返回匹配的记忆列表。",
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = str(args.get("query") or "").strip()
        limit = int(args.get("limit") or 20)

        conn = getattr(ctx, "db", None)
        if conn is None:
            return ToolResult(False, "", "数据库不可用")

        try:
            if query:
                like = f"%{query}%"
                rows = conn.execute(
                    "SELECT * FROM memories WHERE theme LIKE ? OR content LIKE ? ORDER BY updated_at DESC LIMIT ?",
                    (like, like, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()

            columns = ["id", "theme", "content", "created_at", "updated_at"]
            memories = [dict(zip(columns, r)) for r in rows]
            result = {"total": len(memories), "query": query or "(all)", "memories": memories}
            return ToolResult(True, _ok(result), display={"kind": "json", "data": result})
        except Exception as exc:
            return ToolResult(False, "", f"查询失败: {exc}")


def register(registry):
    registry.register(MergeMemoriesTool())
    registry.register(ReadMemoriesTool())