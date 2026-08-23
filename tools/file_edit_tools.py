"""SonettoHere file_edit tool — migrated to SuperAgent plugin.

- file_edit: 文件精确编辑（old_string 精确替换、多笔编辑、行范围读取、正则搜索）
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict

from app.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from app.tools.workspace import resolve_in_workspace, PathEscapeError


def _ok(data: dict) -> str:
    return json.dumps({"success": True, "data": data}, ensure_ascii=False)


def _err(message: str) -> str:
    return json.dumps({"success": False, "error": message}, ensure_ascii=False)


class FileEditTool(Tool):
    spec = ToolSpec(
        name="file_edit",
        description=(
            "文件精确编辑：读取文件行范围、精确字符串替换(old_string)、多笔编辑(multi_edit)、正则搜索。"
            "类似 Claude Code Edit 模式，支持精确匹配替换。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["edit", "multi_edit", "read", "search"],
                    "description": "操作类型",
                },
                "file_path": {"type": "string", "description": "项目内文件相对路径"},
                "old_string": {"type": "string", "description": "要被替换的原文（edit 操作）"},
                "new_string": {"type": "string", "description": "替换后的新内容（edit 操作）"},
                "replace_all": {"type": "boolean", "description": "替换所有匹配项，默认 false"},
                "edits": {
                    "type": "string",
                    "description": "multi_edit 的 JSON 编辑列表：[{old_string, new_string, replace_all}]",
                },
                "offset": {"type": "integer", "description": "读取起始行号（read 操作），0=从头"},
                "limit": {"type": "integer", "description": "读取行数（read 操作），0=全部"},
                "pattern": {"type": "string", "description": "搜索模式，支持正则（search 操作）"},
                "case_insensitive": {"type": "boolean", "description": "搜索忽略大小写，默认 false"},
            },
            "required": ["operation", "file_path"],
        },
        doc=(
            "# file_edit\n\n"
            "文件精确编辑工具。\n"
            "- edit: 精确字符串替换，old_string 必须完全匹配（含空白/缩排/换行）\n"
            "- multi_edit: 批量编辑，传入 JSON 编辑列表\n"
            "- read: 按行范围读取文件\n"
            "- search: 正则搜索文件内容"
        ),
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        operation = str(args.get("operation") or "")
        file_path = str(args.get("file_path") or "")

        if not operation:
            return ToolResult(False, "", "operation 必填: edit / multi_edit / read / search")
        if not file_path:
            return ToolResult(False, "", "file_path 不能为空")

        try:
            p = resolve_in_workspace(ctx.workspace_root, file_path)
        except PathEscapeError as exc:
            return ToolResult(False, "", str(exc))

        if not p.exists():
            return ToolResult(False, "", f"文件不存在: {file_path}")
        if not p.is_file():
            return ToolResult(False, "", f"不是文件: {file_path}")

        try:
            if operation == "read":
                return self._read(p, int(args.get("offset") or 0), int(args.get("limit") or 0))
            elif operation == "edit":
                return self._edit(p, str(args.get("old_string") or ""), str(args.get("new_string") or ""), bool(args.get("replace_all", False)))
            elif operation == "multi_edit":
                return self._multi_edit(p, str(args.get("edits") or ""))
            elif operation == "search":
                return self._search(p, str(args.get("pattern") or ""), bool(args.get("case_insensitive", False)))
            else:
                return ToolResult(False, "", f"未知操作: {operation}")
        except Exception as exc:
            return ToolResult(False, "", str(exc))

    def _read(self, p, offset: int, limit: int) -> ToolResult:
        with open(p, "r", encoding="utf-8") as f:
            lines = f.readlines()
        total = len(lines)
        start = max(0, offset)
        end = start + limit if limit > 0 else total
        selected = lines[start:end]
        result = {
            "file_path": str(p),
            "total_lines": total,
            "offset": start,
            "limit": end - start,
            "lines": [{"num": start + i + 1, "content": l.rstrip("\n\r")} for i, l in enumerate(selected)],
            "content": "".join(selected),
        }
        return ToolResult(True, _ok(result), display={"kind": "json", "data": result})

    def _edit(self, p, old_string: str, new_string: str, replace_all: bool) -> ToolResult:
        if not old_string:
            return ToolResult(False, "", "edit 操作需要提供 old_string")
        with open(p, "r", encoding="utf-8") as f:
            content = f.read()
        count = content.count(old_string)
        if count == 0:
            return ToolResult(False, "", "未找到匹配的 old_string")
        if count > 1 and not replace_all:
            return ToolResult(False, "", f"old_string 有 {count} 处匹配。提供更多上下文或设置 replace_all=true")
        new_content = content.replace(old_string, new_string, -1 if replace_all else 1)
        with open(p, "w", encoding="utf-8") as f:
            f.write(new_content)
        replaced = count if replace_all else 1
        result = {"file_path": str(p), "replaced_count": replaced, "replace_all": replace_all, "message": f"已替换 {replaced} 处"}
        return ToolResult(True, _ok(result), display={"kind": "json", "data": result})

    def _multi_edit(self, p, edits_json: str) -> ToolResult:
        if not edits_json:
            return ToolResult(False, "", "multi_edit 需要 edits 参数")
        try:
            edit_list = json.loads(edits_json)
        except (json.JSONDecodeError, TypeError) as exc:
            return ToolResult(False, "", f"edits JSON 解析失败: {exc}")
        if not isinstance(edit_list, list) or not edit_list:
            return ToolResult(False, "", "edits 应为非空 JSON 数组")
        with open(p, "r", encoding="utf-8") as f:
            content = f.read()
        results = []
        for i, edit in enumerate(edit_list):
            old = edit.get("old_string", "")
            new = edit.get("new_string", "")
            all_ = edit.get("replace_all", False)
            if not old:
                results.append({"index": i, "status": "error", "message": "old_string 为空"})
                continue
            cnt = content.count(old)
            if cnt == 0:
                results.append({"index": i, "status": "error", "message": "未找到匹配"})
                continue
            if cnt > 1 and not all_:
                results.append({"index": i, "status": "error", "message": f"有 {cnt} 处匹配，需 replace_all=true"})
                continue
            content = content.replace(old, new, -1 if all_ else 1)
            results.append({"index": i, "status": "ok", "replaced_count": cnt if all_ else 1})
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        success = sum(1 for r in results if r["status"] == "ok")
        result = {"file_path": str(p), "total_edits": len(edit_list), "success_count": success, "failed_count": len(edit_list) - success, "results": results}
        return ToolResult(True, _ok(result), display={"kind": "json", "data": result})

    def _search(self, p, pattern: str, case_insensitive: bool) -> ToolResult:
        if not pattern:
            return ToolResult(False, "", "search 需要 pattern 参数")
        flags = re.MULTILINE
        if case_insensitive:
            flags |= re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            return ToolResult(False, "", f"正则表达式错误: {exc}")
        with open(p, "r", encoding="utf-8") as f:
            lines = f.readlines()
        matches = []
        for i, line in enumerate(lines):
            for m in regex.finditer(line.rstrip("\n\r")):
                matches.append({"line_num": i + 1, "column": m.start() + 1, "match": m.group()})
        result = {"file_path": str(p), "pattern": pattern, "total_matches": len(matches), "matches": matches[:100]}
        return ToolResult(True, _ok(result), display={"kind": "json", "data": result})


def register(registry):
    registry.register(FileEditTool())