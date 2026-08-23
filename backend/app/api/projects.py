"""Project directory management.

Projects are persisted local directories. Selecting a project writes its path
to `workspace_root`, so all built-in file and command tools remain sandboxed
to that directory by the existing workspace guard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.core.config import default_workspace_root
from app.storage import dao

router = APIRouter(prefix="/api/projects", tags=["projects"])


def _db(request: Request):
    return request.app.state.db


def _normalize_project_path(raw: str) -> str:
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise HTTPException(status_code=400, detail="project path does not exist")
    if not path.is_dir():
        raise HTTPException(status_code=400, detail="project path is not a directory")
    return str(path)


class ProjectCreate(BaseModel):
    path: str
    name: Optional[str] = None


class ProjectRename(BaseModel):
    name: str


def _with_current(conn, project: dict) -> dict:
    current_id = dao.get_setting(conn, "current_project_id")
    return {**project, "current": project["id"] == current_id}


@router.post("/pick-folder")
def pick_folder(request: Request):
    """Open the desktop shell's native folder dialog.

    Returns ``available: False`` when not running inside the pywebview shell
    (e.g. the dev browser), so the frontend can fall back to manual path entry.
    A sync def so FastAPI runs the blocking dialog in a threadpool.
    """
    picker = getattr(request.app.state, "pick_folder", None)
    if picker is None:
        return {"available": False, "path": None}
    try:
        path = picker()
    except Exception as exc:  # noqa: BLE001 — degrade to manual entry on any failure
        return {"available": False, "path": None, "error": str(exc)}
    return {"available": True, "path": path}


@router.get("")
async def list_projects(request: Request):
    conn = _db(request)
    current_id = dao.get_setting(conn, "current_project_id")
    return {
        "current_project_id": current_id,
        "projects": [_with_current(conn, p) for p in dao.list_projects(conn)],
    }


@router.post("")
async def create_project(body: ProjectCreate, request: Request):
    conn = _db(request)
    path = _normalize_project_path(body.path)
    name = (body.name or Path(path).name or path).strip()
    project = dao.create_project(conn, name=name, path=path)
    dao.set_setting(conn, "current_project_id", project["id"])
    dao.set_setting(conn, "workspace_root", project["path"])
    dao.touch_project(conn, project["id"])
    return _with_current(conn, dao.get_project(conn, project["id"]))  # type: ignore[arg-type]


@router.put("/{project_id}/select")
async def select_project(project_id: str, request: Request):
    conn = _db(request)
    project = dao.get_project(conn, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    _normalize_project_path(project["path"])
    dao.set_setting(conn, "current_project_id", project["id"])
    dao.set_setting(conn, "workspace_root", project["path"])
    project = dao.touch_project(conn, project["id"])
    return _with_current(conn, project)  # type: ignore[arg-type]


@router.patch("/{project_id}")
async def rename_project(project_id: str, body: ProjectRename, request: Request):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="project name cannot be empty")
    project = dao.rename_project(_db(request), project_id, name)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    return _with_current(_db(request), project)


@router.delete("/{project_id}")
async def delete_project(project_id: str, request: Request):
    conn = _db(request)
    if not dao.get_project(conn, project_id):
        raise HTTPException(status_code=404, detail="project not found")
    was_current = dao.get_setting(conn, "current_project_id") == project_id
    dao.delete_project(conn, project_id)
    if was_current:
        remaining = dao.list_projects(conn)
        if remaining:
            next_project = remaining[0]
            dao.set_setting(conn, "current_project_id", next_project["id"])
            dao.set_setting(conn, "workspace_root", next_project["path"])
        else:
            dao.set_setting(conn, "current_project_id", None)
            dao.set_setting(conn, "workspace_root", default_workspace_root())
    return {"ok": True}
