"""Provider balance check API — migrated from SonettoHere."""

from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException, Request

from app.core import config as app_config

router = APIRouter(prefix="/api/balance", tags=["balance"])


@router.get("/deepseek")
async def get_deepseek_balance(request: Request):
    """查询 DeepSeek API 账户余额。"""
    conn = request.app.state.db
    providers = [p for p in app_config.load_provider_records(conn) if p.enabled]

    # Find DeepSeek provider
    deepseek = None
    for p in providers:
        if "deepseek" in (p.id + p.label).lower():
            deepseek = p
            break

    if not deepseek:
        raise HTTPException(status_code=400, detail="DeepSeek provider not configured")

    api_key = app_config._resolve_provider_api_key(deepseek.id)
    if not api_key:
        raise HTTPException(status_code=400, detail="DeepSeek API key not set")

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://api.deepseek.com/user/balance",
                headers={"Accept": "application/json", "Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="DeepSeek API timeout")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"DeepSeek API error: {exc}")


@router.get("/{provider_id}")
async def get_provider_balance(provider_id: str, request: Request):
    """查询指定 provider 的余额（通用）。"""
    conn = request.app.state.db
    providers = app_config.load_provider_records(conn)
    provider = next((p for p in providers if p.id == provider_id), None)

    if not provider:
        raise HTTPException(status_code=404, detail=f"Provider {provider_id} not found")

    api_key = app_config._resolve_provider_api_key(provider.id)
    if not api_key:
        raise HTTPException(status_code=400, detail=f"API key for {provider_id} not set")

    # Try common balance endpoints
    balance_urls = {
        "deepseek": "https://api.deepseek.com/user/balance",
        "openai": "https://api.openai.com/v1/dashboard/billing/credit_grants",
    }

    url = balance_urls.get(provider_id)
    if not url:
        # Try to construct from base_url
        base = provider.base_url.rstrip("/")
        url = f"{base}/user/balance"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                url,
                headers={"Accept": "application/json", "Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail=f"{provider_id} API timeout")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{provider_id} API error: {exc}")