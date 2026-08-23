"""Long-term memory inspection API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.agent.memory import format_memory_narrative
from app.storage import dao

router = APIRouter(prefix="/api/memories", tags=["memories"])


def _db(request: Request):
    return request.app.state.db


class MemoryCreate(BaseModel):
    theme: str
    content: str


class MemoryUpdate(BaseModel):
    theme: str | None = None
    content: str | None = None


@router.get("")
async def list_memories(request: Request):
    conn = _db(request)
    return {"memories": dao.list_memories(conn), "narrative": format_memory_narrative(conn)}


@router.post("")
async def create_memory(body: MemoryCreate, request: Request):
    theme = body.theme.strip() or "偏好"
    content = body.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="content is required")
    return dao.add_memory(_db(request), theme, content)


@router.put("/{memory_id}")
async def update_memory(memory_id: str, body: MemoryUpdate, request: Request):
    content = body.content.strip() if body.content is not None else None
    if body.content is not None and not content:
        raise HTTPException(status_code=400, detail="content is required")
    memory = dao.update_memory(
        _db(request),
        memory_id,
        theme=body.theme.strip() if body.theme is not None else None,
        content=content,
    )
    if memory is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return memory


@router.delete("/{memory_id}")
async def delete_memory(memory_id: str, request: Request):
    if not dao.delete_memory(_db(request), memory_id):
        raise HTTPException(status_code=404, detail="memory not found")
    return {"ok": True}


@router.delete("")
async def clear_memories(request: Request):
    dao.clear_memories(_db(request))
    return {"ok": True}
