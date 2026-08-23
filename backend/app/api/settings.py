"""Settings REST API (M2).

Non-secret settings persist in the SQLite `settings` table. The API key is
written to the OS keyring via core.secrets and is NEVER returned to the client
(only a boolean `has_api_key`).
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.core import config, secrets
from app.storage import dao

router = APIRouter(prefix="/api/settings", tags=["settings"])


def _db(request: Request):
    return request.app.state.db


class SettingsUpdate(BaseModel):
    base_url: Optional[str] = None
    model: Optional[str] = None
    active_provider_id: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    context_max_messages: Optional[int] = None
    context_strategy: Optional[str] = None
    context_max_tokens: Optional[int] = None
    max_tool_iterations: Optional[int] = None
    max_tool_arg_len: Optional[int] = None
    max_tool_result_len: Optional[int] = None
    max_tool_text_len: Optional[int] = None
    api_timeout: Optional[int] = None
    turn_timeout: Optional[int] = None
    theme: Optional[str] = None
    workspace_root: Optional[str] = None
    tools_enabled: Optional[bool] = None
    disabled_tools: Optional[List[str]] = None
    memory_enabled: Optional[bool] = None


class ApiKeyUpdate(BaseModel):
    api_key: str


def _effective(conn) -> dict:
    s = config.load_settings(conn)
    return {
        "base_url": s.provider.base_url,
        "model": s.provider.model,
        "active_provider_id": s.active_provider_id,
        "providers": [p.__dict__ for p in config.load_provider_records(conn)],
        "temperature": s.temperature,
        "top_p": s.top_p,
        "max_tokens": s.max_tokens,
        "system_prompt": s.system_prompt,
        "context_max_messages": s.context_max_messages,
        "context_strategy": s.context_strategy,
        "context_max_tokens": s.context_max_tokens,
        "max_tool_iterations": s.max_tool_iterations,
        "max_tool_arg_len": s.max_tool_arg_len,
        "max_tool_result_len": s.max_tool_result_len,
        "max_tool_text_len": s.max_tool_text_len,
        "api_timeout": s.api_timeout,
        "turn_timeout": s.turn_timeout,
        "theme": s.theme,
        "workspace_root": s.workspace_root,
        "current_project_id": dao.get_setting(conn, "current_project_id"),
        "tools_enabled": s.tools_enabled,
        "disabled_tools": list(s.disabled_tools),
        "memory_enabled": s.memory_enabled,
        "has_api_key": secrets.has_api_key(),
    }


@router.get("")
async def get_settings(request: Request):
    return _effective(_db(request))


@router.put("")
async def update_settings(body: SettingsUpdate, request: Request):
    conn = _db(request)
    for key, value in body.model_dump(exclude_unset=True).items():
        dao.set_setting(conn, key, value)
    return _effective(conn)


@router.put("/api-key")
async def set_api_key(body: ApiKeyUpdate):
    secrets.set_api_key(body.api_key)
    return {"ok": True, "has_api_key": True}


@router.delete("/api-key")
async def clear_api_key():
    secrets.delete_api_key()
    return {"ok": True, "has_api_key": False}
