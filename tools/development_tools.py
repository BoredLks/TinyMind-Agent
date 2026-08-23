"""SonettoHere development tools — migrated to SuperAgent plugin.

Pure Python (no external dependencies), provides:
- code_quality: Python 代码质量分析
- syntax_check: 代码语法检查
- unit_test_gen: 单元测试生成提示
- debug_helper: 调试助手
"""

from __future__ import annotations

import ast
import json
import sys
import textwrap
from typing import Any, Dict

# SuperAgent imports
from app.tools.base import Tool, ToolContext, ToolResult, ToolSpec


def _ok(data: dict) -> str:
    return json.dumps({"success": True, "data": data}, ensure_ascii=False)


def _err(message: str) -> str:
    return json.dumps({"success": False, "error": message}, ensure_ascii=False)


class CodeQualityTool(Tool):
    spec = ToolSpec(
        name="code_quality",
        description=(
            "分析 Python 代码质量：复杂度（函数数量/行数）、可维护性（注释率/命名规范）、重复代码检测。"
            "传入 code 字符串或 file_path（项目内相对路径）。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "要分析的 Python 代码（与 file_path 二选一）"},
                "file_path": {"type": "string", "description": "项目内 Python 文件路径（与 code 二选一）"},
                "analysis_type": {
                    "type": "string",
                    "enum": ["complexity", "maintainability", "duplication", "all"],
                    "description": "分析类型，默认 all",
                },
            },
        },
        doc="# code_quality\n\n分析 Python 代码质量。支持 complexity/maintainability/duplication/all 四种分析类型。",
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        code = str(args.get("code") or "")
        file_path = str(args.get("file_path") or "")
        analysis_type = str(args.get("analysis_type") or "all")

        if file_path and not code:
            from pathlib import Path
            from app.tools.workspace import resolve_in_workspace, PathEscapeError
            try:
                p = resolve_in_workspace(ctx.workspace_root, file_path)
            except PathEscapeError as exc:
                return ToolResult(False, "", str(exc))
            if not p.is_file():
                return ToolResult(False, "", f"文件不存在: {file_path}")
            try:
                code = p.read_text(encoding="utf-8")
            except Exception as exc:
                return ToolResult(False, "", f"读取失败: {exc}")

        if not code.strip():
            return ToolResult(False, "", "必须提供 code 或 file_path")

        try:
            result: dict = {}
            if analysis_type in ("complexity", "all"):
                result["complexity"] = _analyze_complexity(code)
            if analysis_type in ("maintainability", "all"):
                result["maintainability"] = _analyze_maintainability(code)
            if analysis_type in ("duplication", "all"):
                result["duplication"] = _analyze_duplication(code)
            return ToolResult(True, _ok(result), display={"kind": "json", "data": result})
        except SyntaxError as exc:
            return ToolResult(False, "", f"代码语法错误: {exc}")
        except Exception as exc:
            return ToolResult(False, "", f"分析失败: {exc}")


def _analyze_complexity(code: str) -> dict:
    tree = ast.parse(code)
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append({
                "name": node.name,
                "line": node.lineno,
                "endline": getattr(node, "end_lineno", node.lineno),
            })
    total_lines = len(code.split("\n"))
    avg_len = 0
    if functions:
        total = sum(f["endline"] - f["line"] + 1 for f in functions)
        avg_len = round(total / len(functions), 1)
    return {
        "total_lines": total_lines,
        "function_count": len(functions),
        "avg_function_length": avg_len,
        "functions": functions,
    }


def _analyze_maintainability(code: str) -> dict:
    lines = code.split("\n")
    comment_lines = sum(1 for line in lines if line.strip().startswith("#"))
    comment_ratio = round(comment_lines / len(lines), 3) if lines else 0

    tree = ast.parse(code)
    snake = 0
    camel = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if "_" in node.name and node.name.islower():
                snake += 1
            elif node.name[0].islower() and any(c.isupper() for c in node.name):
                camel += 1

    score = 30
    if 0.1 <= comment_ratio <= 0.3:
        score += 40
    elif 0.05 <= comment_ratio < 0.1:
        score += 20
    elif comment_ratio > 0.3:
        score += 30
    if snake > camel:
        score += 30
    elif camel > 0:
        score += 15

    return {
        "comment_ratio": comment_ratio,
        "snake_case_count": snake,
        "camel_case_count": camel,
        "maintainability_score": min(score, 100),
    }


def _analyze_duplication(code: str) -> dict:
    lines = code.split("\n")
    counts: dict[str, int] = {}
    for line in lines:
        s = line.strip()
        if s and not s.startswith("#"):
            counts[s] = counts.get(s, 0) + 1
    dups = [{"line": ln, "count": c} for ln, c in counts.items() if c > 1]
    ratio = round(len(dups) / len(lines), 3) if lines else 0
    return {"duplicate_lines": len(dups), "duplicate_ratio": ratio, "duplicates": dups[:10]}


class SyntaxCheckTool(Tool):
    spec = ToolSpec(
        name="syntax_check",
        description="检查 Python 代码的语法错误。返回语法错误位置和描述，无错误则返回 success。",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "要检查的 Python 代码"},
                "file_path": {"type": "string", "description": "项目内文件路径（与 code 二选一）"},
            },
        },
        doc="# syntax_check\n\n检查 Python 代码语法，返回错误位置。纯本地操作，无外部依赖。",
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        code = str(args.get("code") or "")
        file_path = str(args.get("file_path") or "")

        if file_path and not code:
            from pathlib import Path
            from app.tools.workspace import resolve_in_workspace, PathEscapeError
            try:
                p = resolve_in_workspace(ctx.workspace_root, file_path)
            except PathEscapeError as exc:
                return ToolResult(False, "", str(exc))
            if not p.is_file():
                return ToolResult(False, "", f"文件不存在: {file_path}")
            try:
                code = p.read_text(encoding="utf-8")
            except Exception as exc:
                return ToolResult(False, "", f"读取失败: {exc}")

        if not code.strip():
            return ToolResult(False, "", "必须提供 code 或 file_path")

        try:
            ast.parse(code)
            result = {"valid": True, "message": "语法正确"}
            return ToolResult(True, _ok(result), display={"kind": "json", "data": result})
        except SyntaxError as exc:
            result = {
                "valid": False,
                "error": str(exc),
                "line": exc.lineno,
                "offset": exc.offset,
                "text": (exc.text or "").strip(),
            }
            return ToolResult(True, _ok(result), display={"kind": "json", "data": result})


class DebugHelperTool(Tool):
    spec = ToolSpec(
        name="debug_helper",
        description=(
            "调试助手：分析错误信息并提供调试建议。"
            "输入错误的 traceback 或错误消息，返回可能的原因和修复建议。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "error_message": {"type": "string", "description": "错误信息或 traceback"},
                "code_context": {"type": "string", "description": "相关代码上下文（可选）"},
            },
            "required": ["error_message"],
        },
        doc="# debug_helper\n\n输入错误信息，返回常见错误类型分析和调试建议。纯本地规则匹配，不依赖外部 API。",
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        error_msg = str(args.get("error_message") or "").strip()
        code_ctx = str(args.get("code_context") or "").strip()

        if not error_msg:
            return ToolResult(False, "", "error_message 不能为空")

        error_type = "Unknown"
        suggestions = []

        if "SyntaxError" in error_msg:
            error_type = "SyntaxError"
            suggestions = ["检查括号、引号是否配对", "检查缩进是否一致", "检查是否有中文标点"]
        elif "IndentationError" in error_msg:
            error_type = "IndentationError"
            suggestions = ["统一使用 4 个空格缩进", "检查是否有 tab 和空格混用"]
        elif "NameError" in error_msg:
            error_type = "NameError"
            suggestions = ["检查变量名是否拼写正确", "检查变量是否在使用前已定义", "检查是否缺少 import"]
        elif "TypeError" in error_msg:
            error_type = "TypeError"
            suggestions = ["检查函数参数类型是否正确", "检查是否对 None 做了操作", "检查是否缺少类型转换"]
        elif "KeyError" in error_msg:
            error_type = "KeyError"
            suggestions = ["使用 dict.get() 代替直接取值", "检查键名拼写", "检查数据结构是否正确"]
        elif "AttributeError" in error_msg:
            error_type = "AttributeError"
            suggestions = ["检查对象是否有该属性", "检查对象是否为 None", "检查 import 是否正确"]
        elif "ImportError" in error_msg or "ModuleNotFoundError" in error_msg:
            error_type = "ImportError"
            suggestions = ["检查包是否已安装 (pip install)", "检查模块路径是否正确", "检查是否存在循环导入"]
        elif "FileNotFoundError" in error_msg:
            error_type = "FileNotFoundError"
            suggestions = ["检查文件路径是否正确", "检查文件是否存在", "使用绝对路径或 pathlib"]
        elif "IndexError" in error_msg:
            error_type = "IndexError"
            suggestions = ["检查索引是否越界", "检查列表/元组是否为空", "使用 len() 先检查长度"]
        elif "ValueError" in error_msg:
            error_type = "ValueError"
            suggestions = ["检查传入值是否符合要求", "检查类型转换是否可能失败", "添加输入验证"]
        else:
            suggestions = ["查看完整 traceback 定位问题", "检查最近的代码变更", "搜索错误信息获取更多线索"]

        result = {
            "error_type": error_type,
            "suggestions": suggestions,
            "original_message": error_msg[:500],
        }
        if code_ctx:
            result["code_context"] = code_ctx[:1000]

        return ToolResult(True, _ok(result), display={"kind": "json", "data": result})


class UnitTestRunnerTool(Tool):
    spec = ToolSpec(
        name="unit_test_runner",
        description=(
            "执行 Python 单元测试文件，返回通过/失败/错误统计和详细报告。"
            "支持指定测试类和测试方法。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "test_file": {"type": "string", "description": "项目内测试文件相对路径"},
                "test_class": {"type": "string", "description": "特定测试类名（可选）"},
                "test_method": {"type": "string", "description": "特定测试方法名（可选，需配合 test_class）"},
            },
            "required": ["test_file"],
        },
        doc="# unit_test_runner\n\n执行 Python 单元测试文件，返回通过/失败/错误统计和详细报告。",
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        import importlib.util
        import unittest
        import io

        test_file = str(args.get("test_file") or "").strip()
        test_class = str(args.get("test_class") or "").strip()
        test_method = str(args.get("test_method") or "").strip()

        if not test_file:
            return ToolResult(False, "", "test_file 不能为空")

        # Resolve path via workspace
        from app.tools.workspace import resolve_in_workspace, PathEscapeError
        try:
            p = resolve_in_workspace(ctx.workspace_root, test_file)
        except PathEscapeError as exc:
            return ToolResult(False, "", str(exc))

        if not p.exists():
            return ToolResult(False, "", f"测试文件不存在: {test_file}")

        try:
            spec = importlib.util.spec_from_file_location("test_module", str(p))
            if spec is None or spec.loader is None:
                return ToolResult(False, "", f"无法加载测试文件: {test_file}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            loader = unittest.TestLoader()
            if test_class and test_method:
                tests = loader.loadTestsFromName(f"{test_class}.{test_method}", module)
            elif test_class:
                tests = loader.loadTestsFromTestCase(getattr(module, test_class))
            else:
                tests = loader.loadTestsFromModule(module)

            stream = io.StringIO()
            runner = unittest.TextTestRunner(stream=stream, verbosity=2)
            result = runner.run(tests)

            total = result.testsRun
            failures = len(result.failures)
            errors = len(result.errors)
            skipped = len(result.skipped)

            report = {
                "tests_run": total,
                "failures": failures,
                "errors": errors,
                "skipped": skipped,
                "successful": total - failures - errors - skipped,
                "success_rate": round((total - failures - errors) / total * 100, 1) if total > 0 else 0,
                "output": stream.getvalue()[-2000:],
            }

            if result.failures:
                report["failures_details"] = [
                    {"test": str(t), "message": str(e[0][:200])} for t, e in result.failures[:5]
                ]
            if result.errors:
                report["errors_details"] = [
                    {"test": str(t), "message": str(e[0][:200])} for t, e in result.errors[:5]
                ]

            return ToolResult(True, _ok(report), display={"kind": "json", "data": report})
        except Exception as exc:
            return ToolResult(False, "", f"测试执行失败: {exc}")


def register(registry):
    registry.register(CodeQualityTool())
    registry.register(SyntaxCheckTool())
    registry.register(DebugHelperTool())
    registry.register(UnitTestRunnerTool())
