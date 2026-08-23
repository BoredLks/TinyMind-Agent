"""Configuration loading with layered overrides.

Precedence (low → high):
  1. built-in defaults
  2. environment / backend/.env
  3. persisted settings in the SQLite `settings` table (when a conn is given)

The API key is resolved separately: OS keyring first, then env fallback. It is
never read from the settings table.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

from dotenv import load_dotenv

# backend/.env  (config.py -> core -> app -> backend)
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_PATH)

_DEFAULT_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
_DEFAULT_MODEL = "mimo-v2.5-pro"


def _default_workspace() -> str:
    return os.path.join(os.path.expanduser("~"), "SuperAgent", "workspace")


def default_workspace_root() -> str:
    return _default_workspace()


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    api_key: str
    model: str
    id: str = "default"
    label: str = "默认"


@dataclass(frozen=True)
class ProviderRecord:
    id: str
    label: str
    base_url: str
    model: str
    enabled: bool = True
    has_api_key: bool = False


@dataclass(frozen=True)
class Settings:
    provider: ProviderConfig
    active_provider_id: str = "default"
    temperature: float = 0.7
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    context_max_messages: int = 20
    context_strategy: str = "sliding_window"
    context_max_tokens: int = 0
    max_tool_iterations: int = 30
    max_tool_arg_len: int = 4000
    max_tool_result_len: int = 6000
    max_tool_text_len: int = 8000
    api_timeout: int = 300
    turn_timeout: int = 900
    theme: str = "light"
    workspace_root: str = ""
    tools_enabled: bool = True
    disabled_tools: Tuple[str, ...] = field(default_factory=tuple)
    memory_enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 0


def _resolve_api_key() -> str:
    """Keyring first, then env fallback (M1 dev still works via .env)."""
    try:
        from app.core import secrets

        key = secrets.get_api_key()
        if key:
            return key
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "")


def load_settings(conn: Any = None) -> Settings:
    """Build Settings, overlaying persisted DB settings on top of env defaults."""
    db_over: dict[str, Any] = {}
    if conn is not None:
        from app.storage import dao

        db_over = dao.get_all_settings(conn)

    def val(key: str, default: Any) -> Any:
        return db_over[key] if key in db_over else default

    active_provider_id = str(val("active_provider_id", "default") or "default")
    providers = load_provider_records(conn)
    selected = next((p for p in providers if p.id == active_provider_id and p.enabled), None)
    if selected is None:
        selected = next((p for p in providers if p.enabled), providers[0])
        active_provider_id = selected.id
    base_url = selected.base_url
    model = selected.model
    temperature = float(val("temperature", os.getenv("LLM_TEMPERATURE", "0.7")))
    max_tokens = val("max_tokens", None)
    workspace_root = val("workspace_root", os.getenv("SUPERAGENT_WORKSPACE", _default_workspace()))
    disabled_raw = val("disabled_tools", [])
    disabled = tuple(disabled_raw) if isinstance(disabled_raw, (list, tuple)) else ()

    return Settings(
        provider=ProviderConfig(
            base_url=base_url,
            api_key=_resolve_provider_api_key(selected.id),
            model=model,
            id=selected.id,
            label=selected.label,
        ),
        active_provider_id=active_provider_id,
        temperature=temperature,
        top_p=val("top_p", None),
        max_tokens=int(max_tokens) if max_tokens is not None else None,
        system_prompt=val("system_prompt", None),
        context_max_messages=int(val("context_max_messages", 20)),
        context_strategy=val("context_strategy", "sliding_window"),
        context_max_tokens=int(val("context_max_tokens", 0)),
        max_tool_iterations=int(val("max_tool_iterations", 30)),
        max_tool_arg_len=int(val("max_tool_arg_len", 4000)),
        max_tool_result_len=int(val("max_tool_result_len", 6000)),
        max_tool_text_len=int(val("max_tool_text_len", 8000)),
        api_timeout=int(val("api_timeout", 300)),
        turn_timeout=int(val("turn_timeout", 900)),
        theme=val("theme", "light"),
        workspace_root=workspace_root,
        tools_enabled=bool(val("tools_enabled", True)),
        disabled_tools=disabled,
        memory_enabled=bool(val("memory_enabled", True)),
        host=os.getenv("BACKEND_HOST", "127.0.0.1"),
        port=int(os.getenv("BACKEND_PORT", "0")),
    )


def _resolve_provider_api_key(provider_id: str) -> str:
    if provider_id == "default":
        return _resolve_api_key()
    try:
        from app.core import secrets

        key = secrets.get_provider_api_key(provider_id)
        if key:
            return key
    except Exception:
        pass
    return ""


def load_provider_records(conn: Any = None) -> list[ProviderRecord]:
    db_over: dict[str, Any] = {}
    if conn is not None:
        from app.storage import dao

        db_over = dao.get_all_settings(conn)

    legacy = ProviderRecord(
        id="default",
        label="默认",
        base_url=db_over.get("base_url", os.getenv("OPENAI_BASE_URL", _DEFAULT_BASE_URL)),
        model=db_over.get("model", os.getenv("OPENAI_MODEL", _DEFAULT_MODEL)),
        enabled=True,
        has_api_key=False,
    )
    records = [legacy]
    raw = db_over.get("providers", [])
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("id") or "").strip()
            if not pid or pid == "default":
                continue
            records.append(
                ProviderRecord(
                    id=pid,
                    label=str(item.get("label") or pid),
                    base_url=str(item.get("base_url") or _DEFAULT_BASE_URL),
                    model=str(item.get("model") or _DEFAULT_MODEL),
                    enabled=bool(item.get("enabled", True)),
                    has_api_key=False,
                )
            )
    try:
        from app.core import secrets

        enriched = []
        for record in records:
            has_key = secrets.has_api_key() if record.id == "default" else secrets.has_provider_api_key(record.id)
            enriched.append(ProviderRecord(**{**record.__dict__, "has_api_key": has_key}))
        return enriched
    except Exception:
        return records
