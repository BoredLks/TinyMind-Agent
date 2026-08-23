"""External Python tool plugin loader.

A plugin is a `.py` file in one of the configured tools directories. It can
export either:

- `register(registry)`: called with the ToolRegistry, or
- `TOOLS = [Tool(), ...]`: each tool is registered.

These plugins are local code and run with the same privileges as SuperAgent.
Only place plugins here that you trust.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterable

from .base import Tool
from .registry import ToolRegistry


def ensure_tool_dirs(dirs: Iterable[str]) -> None:
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        readme = Path(directory) / "README.md"
        if not readme.exists():
            readme.write_text(
                "# SuperAgent tool plugins\n\n"
                "Put trusted Python `.py` tool plugins in this directory.\n\n"
                "A plugin may export `register(registry)` or `TOOLS = [Tool(), ...]`.\n"
                "Plugins execute as local Python code, so only install plugins you trust.\n",
                encoding="utf-8",
            )


def load_tool_plugins(registry: ToolRegistry, dirs: Iterable[str]) -> list[str]:
    loaded: list[str] = []
    for directory in dirs:
        if not os.path.isdir(directory):
            continue
        for path in sorted(Path(directory).glob("*.py")):
            if path.name.startswith("_"):
                continue
            module_name = f"superagent_tool_plugin_{path.stem}_{abs(hash(str(path)))}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            module_doc = _module_doc(path, module)
            if callable(getattr(module, "register", None)):
                before = set(registry.names())
                module.register(registry)
                new_names = set(registry.names()) - before
                for n in new_names:
                    registry._external.add(n)
                _attach_docs(registry, module_doc, only_names=new_names)
                loaded.append(str(path))
                continue
            tools = getattr(module, "TOOLS", None)
            if tools:
                for tool in tools:
                    if not isinstance(tool, Tool):
                        raise TypeError(f"{path} exported non-Tool in TOOLS")
                    _attach_doc(tool, module_doc)
                    registry.register(tool, external=True)
                loaded.append(str(path))
    return loaded


def _module_doc(path: Path, module) -> dict[str, str] | str | None:
    docs = getattr(module, "TOOL_DOCS", None)
    if isinstance(docs, str) or isinstance(docs, dict):
        return docs
    doc_path = path.parent / "TOOL.md"
    if doc_path.exists():
        return doc_path.read_text(encoding="utf-8")
    return None


def _attach_doc(tool: Tool, docs: dict[str, str] | str | None) -> None:
    if tool.spec.doc:
        return
    if isinstance(docs, dict):
        doc = docs.get(tool.spec.name)
        if doc:
            tool.spec.doc = doc
    elif isinstance(docs, str):
        tool.spec.doc = docs


def _attach_docs(
    registry: ToolRegistry,
    docs: dict[str, str] | str | None,
    *,
    only_names: set[str],
) -> None:
    if docs is None:
        return
    if isinstance(docs, dict):
        for name, doc in docs.items():
            if name not in only_names:
                continue
            tool = registry.get(name)
            if tool and not tool.spec.doc:
                tool.spec.doc = doc
        return
    for name in only_names:
        tool = registry.get(name)
        if tool and not tool.spec.doc:
            tool.spec.doc = docs
