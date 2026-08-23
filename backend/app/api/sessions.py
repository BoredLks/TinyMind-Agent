"""Session + message REST API (M2).

Sessions are created/renamed/deleted here; messages are written by the chat
WebSocket handler and read back here (history restore) and exported.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from app.storage import dao

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _db(request: Request):
    return request.app.state.db


class SessionCreate(BaseModel):
    title: str = "新会话"
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    project_id: Optional[str] = None


class SessionRename(BaseModel):
    title: str


class UndoBody(BaseModel):
    rounds: int = 1


def _export_dir() -> Path:
    override = os.environ.get("SUPERAGENT_EXPORT_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / "Downloads" / "SuperAgent"


def _safe_export_name(sid: str, format: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", sid).strip("._") or "session"
    return f"{stem}.{format}"


def _export_content(conn, sid: str, format: str) -> tuple[str, str, str]:
    session = dao.get_session(conn, sid)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    messages = dao.list_messages(conn, sid)

    # Filter out tool_event messages — they are internal UI metadata
    export_messages = [m for m in messages if m["role"] != "tool_event"]

    if format == "json":
        content = json.dumps(
            {"session": session, "messages": export_messages}, ensure_ascii=False, indent=2
        )
        return content, "application/json", _safe_export_name(sid, "json")

    if format != "md":
        raise HTTPException(status_code=400, detail="format must be md or json")

    label = {"user": "🧑 User", "assistant": "🤖 Assistant", "system": "⚙️ System"}
    lines = [f"# {session['title']}", ""]
    for m in export_messages:
        lines.append(f"**{label.get(m['role'], m['role'])}**")
        lines.append("")
        lines.append(m["content"])
        lines.append("")
    return "\n".join(lines), "text/markdown", _safe_export_name(sid, "md")


@router.get("")
async def list_sessions(request: Request):
    return dao.list_sessions(_db(request))


@router.post("")
async def create_session(body: SessionCreate, request: Request):
    conn = _db(request)
    project_id = body.project_id or None  # normalize "" -> None (ungrouped)
    if project_id and not dao.get_project(conn, project_id):
        raise HTTPException(status_code=400, detail="project not found")
    return dao.create_session(
        conn, title=body.title, model=body.model,
        system_prompt=body.system_prompt, project_id=project_id,
    )


@router.get("/{sid}")
async def get_session(sid: str, request: Request):
    s = dao.get_session(_db(request), sid)
    if not s:
        raise HTTPException(status_code=404, detail="session not found")
    return s


@router.patch("/{sid}")
async def rename_session(sid: str, body: SessionRename, request: Request):
    s = dao.rename_session(_db(request), sid, body.title)
    if not s:
        raise HTTPException(status_code=404, detail="session not found")
    return s


@router.delete("/{sid}")
async def delete_session(sid: str, request: Request):
    dao.delete_session(_db(request), sid)
    return {"ok": True}


@router.get("/{sid}/messages")
async def get_messages(sid: str, request: Request):
    return dao.list_messages(_db(request), sid)


@router.post("/{sid}/undo")
async def undo_session(sid: str, body: UndoBody, request: Request):
    conn = _db(request)
    if not dao.get_session(conn, sid):
        raise HTTPException(status_code=404, detail="session not found")
    deleted = dao.delete_last_turn(conn, sid, body.rounds)
    # Return all messages including tool_events (frontend handles filtering)
    return {"ok": True, "deleted": deleted, "messages": dao.list_messages(conn, sid)}


@router.get("/{sid}/export")
async def export_session(sid: str, request: Request, format: str = "md"):
    content, media_type, filename = _export_content(_db(request), sid, format)
    return Response(
        content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/{sid}/export-file")
async def export_session_file(sid: str, request: Request, format: str = "md"):
    content, _media_type, filename = _export_content(_db(request), sid, format)
    out_dir = _export_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    out_path.write_text(content, encoding="utf-8")
    return {"ok": True, "path": str(out_path), "filename": filename}
