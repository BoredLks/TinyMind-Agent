"""SonettoHere todo tools — migrated to SuperAgent plugin.

Local SQLite-based todo list (no external API needed):
- todo_add: 添加待办
- todo_list: 列出待办
- todo_complete: 完成待办
- todo_uncomplete: 取消完成
- todo_update: 更新待办
- todo_delete: 删除待办
- todo_query: 查询待办
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict

from app.tools.base import Tool, ToolContext, ToolResult, ToolSpec


def _ok(data: dict) -> str:
    return json.dumps({"success": True, "data": data}, ensure_ascii=False)


def _err(message: str) -> str:
    return json.dumps({"success": False, "error": message}, ensure_ascii=False)


def _ensure_table(conn):
    """Create todos table if it doesn't exist."""
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS todos ("
            "  id TEXT PRIMARY KEY,"
            "  content TEXT NOT NULL,"
            "  project TEXT NOT NULL DEFAULT 'Inbox',"
            "  priority INTEGER NOT NULL DEFAULT 1,"
            "  completed INTEGER NOT NULL DEFAULT 0,"
            "  due_date TEXT,"
            "  created_at TEXT NOT NULL,"
            "  updated_at TEXT NOT NULL,"
            "  completed_at TEXT"
            ")"
        )
        conn.commit()
    except Exception:
        pass


def _now() -> str:
    return datetime.now().isoformat()


def _get_db(ctx: ToolContext):
    conn = getattr(ctx, "db", None)
    if conn is None:
        raise RuntimeError("数据库不可用")
    _ensure_table(conn)
    return conn


class TodoAddTool(Tool):
    spec = ToolSpec(
        name="todo_add",
        description="添加一条待办事项。本地存储，无需外部 API。",
        parameters={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "待办内容"},
                "project": {"type": "string", "description": "项目名，默认 Inbox"},
                "priority": {"type": "integer", "description": "优先级 1-4，1 最高，默认 1"},
                "due_date": {"type": "string", "description": "截止日期，格式 YYYY-MM-DD"},
            },
            "required": ["content"],
        },
        doc="# todo_add\n\n添加待办事项。存储在本地 SQLite 中。",
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        content = str(args.get("content") or "").strip()
        if not content:
            return ToolResult(False, "", "content 不能为空")
        try:
            conn = _get_db(ctx)
            todo_id = uuid.uuid4().hex[:12]
            now = _now()
            conn.execute(
                "INSERT INTO todos (id, content, project, priority, due_date, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (todo_id, content, str(args.get("project") or "Inbox"), int(args.get("priority") or 1), str(args.get("due_date") or None) if args.get("due_date") else None, now, now),
            )
            conn.commit()
            result = {"id": todo_id, "content": content, "project": str(args.get("project") or "Inbox"), "priority": int(args.get("priority") or 1)}
            return ToolResult(True, _ok(result), display={"kind": "json", "data": result})
        except Exception as exc:
            return ToolResult(False, "", f"添加失败: {exc}")


class TodoListTool(Tool):
    spec = ToolSpec(
        name="todo_list",
        description="列出待办事项。可按项目和状态筛选。本地存储，无需外部 API。",
        parameters={
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "按项目筛选（可选）"},
                "completed": {"type": "boolean", "description": "筛选已完成/未完成（可选）"},
                "limit": {"type": "integer", "description": "返回数量限制，默认 50"},
            },
        },
        doc="# todo_list\n\n列出待办事项。",
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            conn = _get_db(ctx)
            query = "SELECT * FROM todos WHERE 1=1"
            params = []
            if args.get("project"):
                query += " AND project = ?"
                params.append(args["project"])
            if args.get("completed") is not None:
                query += " AND completed = ?"
                params.append(1 if args["completed"] else 0)
            query += " ORDER BY priority ASC, created_at DESC"
            query += f" LIMIT {int(args.get('limit') or 50)}"

            rows = conn.execute(query, params).fetchall()
            columns = ["id", "content", "project", "priority", "completed", "due_date", "created_at", "updated_at", "completed_at"]
            todos = [dict(zip(columns, row)) for row in rows]
            for t in todos:
                t["completed"] = bool(t["completed"])
            result = {"total": len(todos), "todos": todos}
            return ToolResult(True, _ok(result), display={"kind": "json", "data": result})
        except Exception as exc:
            return ToolResult(False, "", f"查询失败: {exc}")


class TodoCompleteTool(Tool):
    spec = ToolSpec(
        name="todo_complete",
        description="标记待办事项为已完成。本地存储，无需外部 API。",
        parameters={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "待办 ID"},
            },
            "required": ["id"],
        },
        doc="# todo_complete\n\n完成待办。",
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        todo_id = str(args.get("id") or "").strip()
        if not todo_id:
            return ToolResult(False, "", "id 不能为空")
        try:
            conn = _get_db(ctx)
            now = _now()
            cur = conn.execute("UPDATE todos SET completed = 1, completed_at = ?, updated_at = ? WHERE id = ?", (now, now, todo_id))
            conn.commit()
            if cur.rowcount == 0:
                return ToolResult(False, "", f"未找到待办: {todo_id}")
            result = {"id": todo_id, "completed": True}
            return ToolResult(True, _ok(result), display={"kind": "json", "data": result})
        except Exception as exc:
            return ToolResult(False, "", f"操作失败: {exc}")


class TodoUncompleteTool(Tool):
    spec = ToolSpec(
        name="todo_uncomplete",
        description="取消待办的完成状态。本地存储，无需外部 API。",
        parameters={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "待办 ID"},
            },
            "required": ["id"],
        },
        doc="# todo_uncomplete\n\n取消完成。",
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        todo_id = str(args.get("id") or "").strip()
        if not todo_id:
            return ToolResult(False, "", "id 不能为空")
        try:
            conn = _get_db(ctx)
            now = _now()
            cur = conn.execute("UPDATE todos SET completed = 0, completed_at = NULL, updated_at = ? WHERE id = ?", (now, todo_id))
            conn.commit()
            if cur.rowcount == 0:
                return ToolResult(False, "", f"未找到待办: {todo_id}")
            result = {"id": todo_id, "completed": False}
            return ToolResult(True, _ok(result), display={"kind": "json", "data": result})
        except Exception as exc:
            return ToolResult(False, "", f"操作失败: {exc}")


class TodoUpdateTool(Tool):
    spec = ToolSpec(
        name="todo_update",
        description="更新待办事项的内容、项目、优先级或截止日期。本地存储，无需外部 API。",
        parameters={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "待办 ID"},
                "content": {"type": "string", "description": "新内容（可选）"},
                "project": {"type": "string", "description": "新项目（可选）"},
                "priority": {"type": "integer", "description": "新优先级（可选）"},
                "due_date": {"type": "string", "description": "新截止日期（可选）"},
            },
            "required": ["id"],
        },
        doc="# todo_update\n\n更新待办。",
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        todo_id = str(args.get("id") or "").strip()
        if not todo_id:
            return ToolResult(False, "", "id 不能为空")
        try:
            conn = _get_db(ctx)
            existing = conn.execute("SELECT * FROM todos WHERE id = ?", (todo_id,)).fetchone()
            if not existing:
                return ToolResult(False, "", f"未找到待办: {todo_id}")

            updates = []
            params = []
            if args.get("content"):
                updates.append("content = ?")
                params.append(args["content"])
            if args.get("project"):
                updates.append("project = ?")
                params.append(args["project"])
            if args.get("priority"):
                updates.append("priority = ?")
                params.append(int(args["priority"]))
            if "due_date" in args:
                updates.append("due_date = ?")
                params.append(args["due_date"])

            if not updates:
                return ToolResult(False, "", "没有要更新的字段")

            updates.append("updated_at = ?")
            params.append(_now())
            params.append(todo_id)
            conn.execute(f"UPDATE todos SET {', '.join(updates)} WHERE id = ?", params)
            conn.commit()
            result = {"id": todo_id, "updated": True}
            return ToolResult(True, _ok(result), display={"kind": "json", "data": result})
        except Exception as exc:
            return ToolResult(False, "", f"更新失败: {exc}")


class TodoDeleteTool(Tool):
    spec = ToolSpec(
        name="todo_delete",
        description="删除一条待办事项。本地存储，无需外部 API。",
        parameters={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "待办 ID"},
            },
            "required": ["id"],
        },
        requires_approval=True,
        doc="# todo_delete\n\n删除待办。",
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        todo_id = str(args.get("id") or "").strip()
        if not todo_id:
            return ToolResult(False, "", "id 不能为空")
        try:
            conn = _get_db(ctx)
            cur = conn.execute("DELETE FROM todos WHERE id = ?", (todo_id,))
            conn.commit()
            if cur.rowcount == 0:
                return ToolResult(False, "", f"未找到待办: {todo_id}")
            result = {"id": todo_id, "deleted": True}
            return ToolResult(True, _ok(result), display={"kind": "json", "data": result})
        except Exception as exc:
            return ToolResult(False, "", f"删除失败: {exc}")


class TodoQueryTool(Tool):
    spec = ToolSpec(
        name="todo_query",
        description="按关键词查询待办事项。模糊匹配 content 字段。本地存储，无需外部 API。",
        parameters={
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "搜索关键词"},
                "limit": {"type": "integer", "description": "返回数量限制，默认 20"},
            },
            "required": ["keyword"],
        },
        doc="# todo_query\n\n按关键词模糊查询待办事项。",
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        keyword = str(args.get("keyword") or "").strip()
        limit = int(args.get("limit") or 20)
        if not keyword:
            return ToolResult(False, "", "keyword 不能为空")
        try:
            conn = _get_db(ctx)
            like = f"%{keyword}%"
            rows = conn.execute(
                "SELECT * FROM todos WHERE content LIKE ? ORDER BY priority ASC, created_at DESC LIMIT ?",
                (like, limit),
            ).fetchall()
            columns = ["id", "content", "project", "priority", "completed", "due_date", "created_at", "updated_at", "completed_at"]
            todos = [dict(zip(columns, row)) for row in rows]
            for t in todos:
                t["completed"] = bool(t["completed"])
            result = {"total": len(todos), "keyword": keyword, "todos": todos}
            return ToolResult(True, _ok(result), display={"kind": "json", "data": result})
        except Exception as exc:
            return ToolResult(False, "", f"查询失败: {exc}")


class TodoListProjectsTool(Tool):
    spec = ToolSpec(
        name="todo_list_projects",
        description="列出所有待办事项项目名称。本地存储，无需外部 API。",
        parameters={
            "type": "object",
            "properties": {},
        },
        doc="# todo_list_projects\n\n列出所有待办项目名称。",
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            conn = _get_db(ctx)
            rows = conn.execute(
                "SELECT project, COUNT(*) as count FROM todos GROUP BY project ORDER BY project"
            ).fetchall()
            projects = [{"name": r[0], "count": r[1]} for r in rows]
            result = {"total": len(projects), "projects": projects}
            return ToolResult(True, _ok(result), display={"kind": "json", "data": result})
        except Exception as exc:
            return ToolResult(False, "", f"查询失败: {exc}")


def register(registry):
    registry.register(TodoAddTool())
    registry.register(TodoListTool())
    registry.register(TodoCompleteTool())
    registry.register(TodoUncompleteTool())
    registry.register(TodoUpdateTool())
    registry.register(TodoDeleteTool())
    registry.register(TodoQueryTool())
    registry.register(TodoListProjectsTool())
