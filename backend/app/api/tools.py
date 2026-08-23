"""Tool registry inspection API."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api/tools", tags=["tools"])


@router.get("")
async def list_tools(request: Request):
    registry = request.app.state.registry
    tools = []
    for name in registry.names():
        tool = registry.get(name)
        if tool is None:
            continue
        tools.append(
            {
                "name": tool.spec.name,
                "description": tool.spec.description,
                "requires_approval": tool.spec.requires_approval,
                "has_doc": bool(tool.spec.doc),
                "external": registry.is_external(name),
            }
        )
    return {
        "tools": tools,
        "loaded_plugins": list(getattr(request.app.state, "loaded_tool_plugins", [])),
    }


@router.get("/{name}/doc")
async def get_tool_doc(name: str, request: Request):
    registry = request.app.state.registry
    doc = registry.doc_for(name)
    if doc is None:
        raise HTTPException(status_code=404, detail="tool not found")
    return {"name": name, "doc": doc}


@router.get("/mentions")
async def list_mentions(request: Request, q: str = ""):
    """Return all mentionable items: tools, skills, files/dirs in workspace."""
    items = []

    # Tools
    registry = request.app.state.registry
    for name in registry.names():
        tool = registry.get(name)
        if tool is None:
            continue
        items.append({
            "type": "tool",
            "name": tool.spec.name,
            "label": tool.spec.name,
            "description": tool.spec.description[:80],
        })

    # Skills
    skills = getattr(request.app.state, "skills", None)
    if skills is not None:
        for m in skills.list_metas():
            items.append({
                "type": "skill",
                "name": m.name,
                "label": m.name,
                "description": m.description[:80],
            })

    # Files/dirs in current workspace
    from app.tools.workspace import ensure_workspace
    from app.core.config import load_settings
    conn = getattr(request.app.state, "db", None)
    if conn is not None:
        settings = load_settings(conn)
        workspace = settings.workspace_root
        if workspace:
            try:
                root = Path(ensure_workspace(workspace))
                max_items = 50
                count = 0
                for entry in sorted(root.rglob("*")):
                    if count >= max_items:
                        break
                    if any(part.startswith(".") for part in entry.relative_to(root).parts):
                        continue
                    rel = str(entry.relative_to(root)).replace("\\", "/")
                    items.append({
                        "type": "file" if entry.is_file() else "dir",
                        "name": rel,
                        "label": rel,
                        "description": f"{'文件' if entry.is_file() else '目录'}: {rel}",
                    })
                    count += 1
            except Exception:
                pass

    # Filter by query
    if q:
        q_lower = q.lower()
        items = [i for i in items if q_lower in i["name"].lower() or q_lower in i.get("description", "").lower()]

    return {"items": items[:30]}
