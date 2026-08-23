"""Confirmed Python execution tool."""

from __future__ import annotations

import asyncio
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from .base import Tool, ToolContext, ToolResult, ToolSpec
from .workspace import PathEscapeError, resolve_in_workspace

_DEFAULT_TIMEOUT = 15
_MAX_TIMEOUT = 60


def _python_cmd() -> List[str] | None:
    if not getattr(sys, "frozen", False):
        return [sys.executable]
    for candidate in (["python"], ["py", "-3"], ["python3"]):
        if shutil.which(candidate[0]):
            return candidate
    return None


class RunPythonTool(Tool):
    spec = ToolSpec(
        name="run_python",
        description=(
            "在当前项目目录中执行一段 Python 代码。执行前会暂停并要求用户确认代码。"
            "适合计算、文本处理和小脚本验证；不要用于长期服务或无限循环。[使用前建议 load_tool_doc]"
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "要执行的 Python 代码"},
                "timeout": {
                    "type": "integer",
                    "description": f"超时秒数，默认 {_DEFAULT_TIMEOUT}，最大 {_MAX_TIMEOUT}",
                },
            },
            "required": ["code"],
        },
        requires_approval=False,
        doc=(
            "# run_python\n\n"
            "执行流程：先把代码展示给用户确认，用户同意后在当前项目目录启动 Python 子进程。"
            "工作目录被限制为当前项目目录；脚本文件临时写入 `.superagent_tmp/` 并在执行后删除。"
            "返回 stdout、stderr、exit_code 和耗时状态。不要运行无限循环、后台服务、破坏性文件操作。"
        ),
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        code = str(args.get("code") or "")
        if not code.strip():
            return ToolResult(False, "", "code 不能为空")
        timeout = int(args.get("timeout") or _DEFAULT_TIMEOUT)
        timeout = max(1, min(timeout, _MAX_TIMEOUT))

        if ctx.interact is None:
            return ToolResult(False, "", "当前通道不支持代码执行确认")

        answer = await ctx.interact(
            {
                "tool_name": self.spec.name,
                "mode": "confirm",
                "question": "即将执行以下 Python 代码，是否确认？",
                "options": ["执行", "取消"],
                "code": code,
            }
        )
        approved = False
        if isinstance(answer, dict):
            approved = bool(answer.get("approved")) or answer.get("action") in {"approve", "执行"}
        elif isinstance(answer, str):
            approved = answer in {"approve", "执行", "yes", "true"}
        elif isinstance(answer, bool):
            approved = answer
        if not approved:
            return ToolResult(
                False,
                "",
                "用户取消了 Python 代码执行",
                display={"kind": "python", "code": code, "cancelled": True},
            )

        cmd = _python_cmd()
        if not cmd:
            return ToolResult(False, "", "未找到可用的 Python 解释器")

        try:
            tmp_dir = resolve_in_workspace(ctx.workspace_root, ".superagent_tmp")
        except PathEscapeError as exc:
            return ToolResult(False, "", str(exc))
        tmp_dir.mkdir(parents=True, exist_ok=True)
        script = tmp_dir / f"run_{secrets.token_hex(6)}.py"
        script.write_text(code, encoding="utf-8")

        def _run() -> subprocess.CompletedProcess:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            return subprocess.run(
                [*cmd, str(script)],
                cwd=ctx.workspace_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env=env,
                shell=False,
            )

        try:
            proc = await asyncio.to_thread(_run)
        except subprocess.TimeoutExpired:
            return ToolResult(
                False,
                "",
                f"Python 执行超时（{timeout}s）",
                display={"kind": "python", "code": code, "timeout": timeout},
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(False, "", f"Python 执行失败：{exc}")
        finally:
            try:
                script.unlink(missing_ok=True)
            except OSError:
                pass

        display = {
            "kind": "python",
            "code": code,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
            "timeout": timeout,
        }
        content = (
            f"Python exit_code={proc.returncode}\n"
            f"[stdout]\n{proc.stdout or '(empty)'}\n"
            f"[stderr]\n{proc.stderr or '(empty)'}"
        )
        return ToolResult(
            proc.returncode == 0,
            content,
            None if proc.returncode == 0 else f"Python 退出码 {proc.returncode}",
            display=display,
        )
