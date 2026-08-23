"""Built-in tools for M3.1: read_file, write_file, list_dir, run_command.

All are sandboxed to the workspace. write_file and run_command are
side-effectful and marked requires_approval=True; the agent loop gates them
behind a UI approval.
"""

from __future__ import annotations

import asyncio
import subprocess
from typing import Any, Dict

from .base import Tool, ToolContext, ToolResult, ToolSpec
from .workspace import PathEscapeError, resolve_in_workspace, validate_command_paths

_COMMAND_TIMEOUT = 60


class ReadFileTool(Tool):
    spec = ToolSpec(
        name="read_file",
        description="Read a UTF-8 text file inside the workspace.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "workspace-relative path"}},
            "required": ["path"],
        },
        requires_approval=False,
        doc=(
            "# read_file\n\n"
            "读取当前项目目录内的 UTF-8 文本文件。路径必须是项目内相对路径或解析后仍位于项目内。"
            "适合查看源码、配置、日志片段；大文件应先用 list_dir 或命令确认范围。"
        ),
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            p = resolve_in_workspace(ctx.workspace_root, args["path"])
        except PathEscapeError as exc:
            return ToolResult(False, "", str(exc))
        if not p.is_file():
            return ToolResult(False, "", f"not a file: {args['path']}")
        try:
            content = p.read_text(encoding="utf-8")
            return ToolResult(
                True,
                content,
                display={
                    "kind": "file", "path": args["path"], "chars": len(content), "action": "read",
                    "previewAvailable": True, "previewContent": content,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(False, "", f"read error: {exc}")


class WriteFileTool(Tool):
    spec = ToolSpec(
        name="write_file",
        description="Create or overwrite a UTF-8 text file inside the workspace.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "workspace-relative path"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
        requires_approval=True,
        doc=(
            "# write_file\n\n"
            "创建或覆盖当前项目目录内的 UTF-8 文本文件。此工具会请求用户批准。"
            "只写入用户要求或项目需要的文件，避免把生成产物写到项目外。"
        ),
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            p = resolve_in_workspace(ctx.workspace_root, args["path"])
        except PathEscapeError as exc:
            return ToolResult(False, "", str(exc))
        content = args.get("content", "")
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(False, "", f"write error: {exc}")
        return ToolResult(
            True,
            f"wrote {len(content)} chars to {args['path']}",
            display={
                "kind": "file", "path": args["path"], "chars": len(content), "action": "write",
                "previewAvailable": True, "previewContent": content,
            },
        )


class ListDirTool(Tool):
    spec = ToolSpec(
        name="list_dir",
        description="List entries of a directory inside the workspace.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "workspace-relative path, default '.'"}},
        },
        requires_approval=False,
        doc=(
            "# list_dir\n\n"
            "列出当前项目目录内某个目录的一层子项。目录用 `[dir]` 标记，文件用 `[file]` 标记。"
            "用于探索项目结构，优先传相对路径。"
        ),
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        requested = args.get("path") or "."
        try:
            p = resolve_in_workspace(ctx.workspace_root, requested)
        except PathEscapeError as exc:
            return ToolResult(False, "", str(exc))
        if not p.is_dir():
            return ToolResult(False, "", f"not a directory: {requested}")
        children = sorted(p.iterdir())
        entries = [("[dir] " if c.is_dir() else "[file] ") + c.name for c in children]
        display_entries = [
            {"name": c.name, "type": "dir" if c.is_dir() else "file"}
            for c in children
        ]
        return ToolResult(
            True,
            "\n".join(entries) if entries else "(empty)",
            display={"kind": "directory", "path": requested, "entries": display_entries},
        )


class RunCommandTool(Tool):
    spec = ToolSpec(
        name="run_command",
        description=(
            "Run a Windows PowerShell command in the current project/workspace directory and return "
            "its output. Use paths relative to the workspace; absolute paths outside the workspace "
            "are rejected."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "shell": {
                    "type": "string",
                    "enum": ["powershell", "cmd"],
                    "description": "Shell to use. Defaults to powershell; use cmd for cmd.exe-only commands.",
                },
            },
            "required": ["command"],
        },
        requires_approval=True,
        doc=(
            "# run_command\n\n"
            "在当前项目目录运行命令，默认 shell 为 Windows PowerShell；只有命令确实依赖 cmd.exe 语法时才传 `shell: \"cmd\"`。"
            "命令中的绝对路径会被检查，不能写入或读取项目目录外的位置。此工具会请求用户批准。"
        ),
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = (args.get("command") or "").strip()
        if not command:
            return ToolResult(False, "", "empty command")
        shell = str(args.get("shell") or "powershell").lower()
        if shell not in {"powershell", "cmd"}:
            return ToolResult(False, "", "unsupported shell (use powershell or cmd)")
        try:
            validate_command_paths(ctx.workspace_root, command)
        except PathEscapeError as exc:
            return ToolResult(False, "", str(exc))

        def _run():
            argv = (
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    command,
                ]
                if shell == "powershell"
                else ["cmd.exe", "/d", "/s", "/c", command]
            )
            return subprocess.run(
                argv,
                shell=False,
                cwd=ctx.workspace_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=_COMMAND_TIMEOUT,
            )

        try:
            proc = await asyncio.to_thread(_run)
        except subprocess.TimeoutExpired:
            return ToolResult(False, "", f"command timed out ({_COMMAND_TIMEOUT}s)")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(False, "", f"exec error: {exc}")

        out = proc.stdout or ""
        if proc.stderr:
            out += f"\n[stderr]\n{proc.stderr}"
        body = f"shell={shell}\nexit={proc.returncode}\n{out}".strip()
        return ToolResult(
            proc.returncode == 0,
            body,
            None if proc.returncode == 0 else f"exit {proc.returncode}",
            display={
                "kind": "command",
                "shell": shell,
                "command": command,
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            },
        )
