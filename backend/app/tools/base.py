"""Tool contracts.

A Tool exposes a ToolSpec (name + description + JSON-Schema parameters +
whether it needs user approval) and an async run(args, ctx) -> ToolResult.
The agent loop turns specs into the provider's `tools` schema, executes calls,
and feeds ToolResult.content back to the model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema for the arguments object
    requires_approval: bool = False
    doc: Optional[str] = None


@dataclass
class ToolResult:
    ok: bool
    content: str  # text returned to the model (and shown in the UI card)
    error: Optional[str] = None
    display: Optional[Dict[str, Any]] = None


InteractFn = Callable[[Dict[str, Any]], Awaitable[Any]]


@dataclass
class ToolContext:
    workspace_root: str
    skills: Any = None  # SkillService, for the load_skill tool
    subagent_runner: Any = None  # SubagentRunner, for the dispatch_subagent tool
    registry: Any = None  # ToolRegistry, for the load_tool_doc tool
    interact: Optional[InteractFn] = None  # pause/wait bridge to the UI
    current_call_id: Optional[str] = None
    session_id: Optional[str] = None
    subagent_depth: int = 0
    db: Any = None


class Tool(ABC):
    spec: ToolSpec

    @abstractmethod
    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        raise NotImplementedError
