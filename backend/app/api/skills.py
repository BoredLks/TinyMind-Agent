"""Skills REST API (M3.2): list / detail / enable-disable."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.storage import dao

router = APIRouter(prefix="/api/skills", tags=["skills"])


def _svc(request: Request):
    return request.app.state.skills


def _db(request: Request):
    return request.app.state.db


class SkillToggle(BaseModel):
    enabled: bool


@router.get("")
async def list_skills(request: Request):
    svc = _svc(request)
    states = dao.get_skill_states(_db(request))
    return [
        {
            "name": m.name,
            "description": m.description,
            "source": m.source,
            "enabled": states.get(m.name, True),
        }
        for m in svc.list_metas()
    ]


@router.get("/{name}")
async def get_skill(name: str, request: Request):
    svc = _svc(request)
    meta = svc.get_meta(name)
    if not meta:
        raise HTTPException(status_code=404, detail="skill not found")
    return {
        "name": meta.name,
        "description": meta.description,
        "source": meta.source,
        "enabled": dao.is_skill_enabled(_db(request), name),
        "body": svc.get_body(name),
    }


@router.put("/{name}")
async def toggle_skill(name: str, body: SkillToggle, request: Request):
    svc = _svc(request)
    if not svc.get_meta(name):
        raise HTTPException(status_code=404, detail="skill not found")
    dao.set_skill_enabled(_db(request), name, body.enabled)
    return {"name": name, "enabled": body.enabled}
