"""Tool registry: holds tools, generates the provider schema, honors enable/disable."""

from __future__ import annotations

from typing import Dict, List, Optional

from .base import Tool
from .builtin import ListDirTool, ReadFileTool, RunCommandTool, WriteFileTool
from .interaction import AskUserMultiChoiceTool, AskUserQATool, AskUserSingleChoiceTool
from .python_exec import RunPythonTool
from .sonetto_tools import sonetto_compatible_tools
from .tool_doc import LoadToolDocTool


def default_tools() -> List[Tool]:
    return [
        ReadFileTool(),
        WriteFileTool(),
        ListDirTool(),
        RunCommandTool(),
        LoadToolDocTool(),
        AskUserQATool(),
        AskUserSingleChoiceTool(),
        AskUserMultiChoiceTool(),
        RunPythonTool(),
        *sonetto_compatible_tools(),
    ]


class ToolRegistry:
    def __init__(self, tools: Optional[List[Tool]] = None) -> None:
        self._tools: Dict[str, Tool] = {}
        self._external: set[str] = set()
        for tool in tools if tools is not None else default_tools():
            self.register(tool)

    def register(self, tool: Tool, *, external: bool = False) -> None:
        self._tools[tool.spec.name] = tool
        if external:
            self._external.add(tool.spec.name)

    def is_external(self, name: str) -> bool:
        return name in self._external

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def doc_for(self, name: str) -> Optional[str]:
        tool = self.get(name)
        if tool is None:
            return None
        if tool.spec.doc:
            return tool.spec.doc
        return (
            f"# {tool.spec.name}\n\n"
            f"{tool.spec.description}\n\n"
            "## 参数 JSON Schema\n\n"
            f"```json\n{tool.spec.parameters}\n```"
        )

    def docs_index(self) -> List[dict]:
        return [
            {
                "name": t.spec.name,
                "description": t.spec.description,
                "has_doc": bool(t.spec.doc),
            }
            for t in self._tools.values()
        ]

    def enabled_tools(self, disabled: Optional[List[str]] = None) -> List[Tool]:
        blocked = set(disabled or [])
        return [t for name, t in self._tools.items() if name not in blocked]

    def openai_schema(self, disabled: Optional[List[str]] = None) -> List[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.spec.name,
                    "description": t.spec.description,
                    "parameters": t.spec.parameters,
                },
            }
            for t in self.enabled_tools(disabled)
        ]
