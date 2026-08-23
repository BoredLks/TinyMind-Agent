"""Multiple LLM provider configuration and health checks."""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from openai import AsyncOpenAI
from pydantic import BaseModel

from app.core import config, secrets
from app.storage import dao

router = APIRouter(prefix="/api/providers", tags=["providers"])


class ProviderUpsert(BaseModel):
    id: str
    label: str
    base_url: str
    model: str = "default"
    api_key: Optional[str] = None
    enabled: bool = True


class ProviderUpdate(BaseModel):
    label: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    enabled: Optional[bool] = None


class ProviderTestBody(BaseModel):
    id: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None


class ProviderDiscoverBody(BaseModel):
    id: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None


def _db(request: Request):
    return request.app.state.db


def _stored(conn) -> list[dict]:
    raw = dao.get_setting(conn, "providers", [])
    return raw if isinstance(raw, list) else []


def _save(conn, providers: list[dict]) -> None:
    dao.set_setting(conn, "providers", providers)


def _public(conn) -> dict:
    active = dao.get_setting(conn, "active_provider_id", "default")
    return {
        "active_provider_id": active,
        "providers": [p.__dict__ for p in config.load_provider_records(conn)],
    }


def _find(records: list[dict], provider_id: str) -> dict | None:
    return next((p for p in records if p.get("id") == provider_id), None)


def _connection_from_body(conn, body: ProviderDiscoverBody | ProviderTestBody) -> tuple[str, str]:
    if body.id:
        record = next((p for p in config.load_provider_records(conn) if p.id == body.id), None)
        if record is None:
            raise HTTPException(status_code=404, detail="provider not found")
        return (
            body.base_url or record.base_url,
            body.api_key or config._resolve_provider_api_key(record.id),  # noqa: SLF001
        )
    return body.base_url or "", body.api_key or ""


@router.get("")
async def list_providers(request: Request):
    return _public(_db(request))


@router.post("")
async def create_provider(body: ProviderUpsert, request: Request):
    conn = _db(request)
    if body.id == "default":
        raise HTTPException(status_code=400, detail="default provider is managed in settings")
    providers = _stored(conn)
    if _find(providers, body.id):
        raise HTTPException(status_code=409, detail="provider already exists")
    providers.append(
        {
            "id": body.id,
            "label": body.label,
            "base_url": body.base_url,
            "model": body.model,
            "enabled": body.enabled,
        }
    )
    if body.api_key:
        secrets.set_provider_api_key(body.id, body.api_key)
    _save(conn, providers)
    return _public(conn)


@router.put("/{provider_id}")
async def update_provider(provider_id: str, body: ProviderUpdate, request: Request):
    conn = _db(request)
    if provider_id == "default":
        patch = {}
        if body.base_url is not None:
            patch["base_url"] = body.base_url
        if body.model is not None:
            patch["model"] = body.model
        for key, value in patch.items():
            dao.set_setting(conn, key, value)
        if body.api_key:
            secrets.set_api_key(body.api_key)
        return _public(conn)

    providers = _stored(conn)
    provider = _find(providers, provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail="provider not found")
    for key in ("label", "base_url", "model", "enabled"):
        value = getattr(body, key)
        if value is not None:
            provider[key] = value
    if body.api_key:
        secrets.set_provider_api_key(provider_id, body.api_key)
    _save(conn, providers)
    return _public(conn)


@router.delete("/{provider_id}")
async def delete_provider(provider_id: str, request: Request):
    if provider_id == "default":
        raise HTTPException(status_code=400, detail="default provider cannot be deleted")
    conn = _db(request)
    providers = [p for p in _stored(conn) if p.get("id") != provider_id]
    _save(conn, providers)
    secrets.delete_provider_api_key(provider_id)
    if dao.get_setting(conn, "active_provider_id", "default") == provider_id:
        dao.set_setting(conn, "active_provider_id", "default")
    return _public(conn)


@router.put("/{provider_id}/active")
async def activate_provider(provider_id: str, request: Request):
    conn = _db(request)
    if not any(p.id == provider_id for p in config.load_provider_records(conn)):
        raise HTTPException(status_code=404, detail="provider not found")
    dao.set_setting(conn, "active_provider_id", provider_id)
    return _public(conn)


@router.post("/test")
async def test_provider(body: ProviderTestBody, request: Request):
    conn = _db(request)
    if body.id:
        record = next((p for p in config.load_provider_records(conn) if p.id == body.id), None)
        if record is None:
            raise HTTPException(status_code=404, detail="provider not found")
        base_url = body.base_url or record.base_url
        api_key = body.api_key or config._resolve_provider_api_key(record.id)  # noqa: SLF001
        model = body.model or record.model
    else:
        base_url = body.base_url or ""
        api_key = body.api_key or ""
        model = body.model or config._DEFAULT_MODEL  # noqa: SLF001
    if not base_url or not api_key:
        raise HTTPException(status_code=400, detail="base_url and api_key are required")

    started = time.perf_counter()
    try:
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0,
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "latency_ms": round((time.perf_counter() - started) * 1000), "detail": str(exc)}
    return {"ok": True, "latency_ms": round((time.perf_counter() - started) * 1000), "detail": "ok"}


@router.post("/discover-models")
async def discover_models(body: ProviderDiscoverBody, request: Request):
    conn = _db(request)
    base_url, api_key = _connection_from_body(conn, body)
    if not base_url or not api_key:
        raise HTTPException(status_code=400, detail="base_url and api_key are required")
    started = time.perf_counter()
    try:
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        models = await client.models.list()
        names = sorted(
            {
                str(getattr(model, "id", "")).strip()
                for model in getattr(models, "data", [])
                if str(getattr(model, "id", "")).strip()
            }
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "models": [],
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "detail": str(exc),
        }
    return {
        "ok": True,
        "models": names,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "detail": "ok",
    }
