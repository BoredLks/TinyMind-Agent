"""FastAPI application assembly.

create_app() opens the SQLite connection (app.state.db), initializes the
schema, and wires the model adapter to read persisted settings + keyring.
`app.state.adapter_factory` is overridable by tests.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.agent.model.openai_compatible import OpenAICompatibleAdapter
from app.agent.memory import MemoryWorker
from app.agent.personas import ensure_external_persona_dirs
from app.api import chat_ws, health, memories, providers, sessions
from app.api import projects as projects_api
from app.api import settings as settings_api
from app.api import skills as skills_api
from app.api import tools as tools_api
from app.api.balance import router as balance_router
from app.core.config import load_settings
from app.core import secrets as app_secrets
from app.core.resources import (
    builtin_skills_dir,
    external_skills_dirs,
    external_tools_dirs,
    frontend_dist_dir,
)
from app.skills.loader import SkillService, default_user_skills_dir
from app.storage import db
from app.tools.plugins import ensure_tool_dirs, load_tool_plugins
from app.tools.registry import ToolRegistry
from app.tools.skill_tool import LoadSkillTool
from app.tools.subagent_tool import DispatchSubagentTool


def build_adapter(conn, provider_id: Optional[str] = None) -> OpenAICompatibleAdapter:
    from app.core import config

    s = load_settings(conn)
    provider = s.provider
    if provider_id and provider_id != provider.id:
        record = next((p for p in config.load_provider_records(conn) if p.id == provider_id and p.enabled), None)
        if record is not None:
            provider = config.ProviderConfig(
                id=record.id,
                label=record.label,
                base_url=record.base_url,
                model=record.model,
                api_key=config._resolve_provider_api_key(record.id),  # noqa: SLF001
            )
    has_key = bool(provider.api_key)
    logger.info(
        "Building adapter: provider=%s, model=%s, base_url=%s, has_api_key=%s",
        provider.id, provider.model, provider.base_url, has_key,
    )
    if not has_key:
        logger.warning(
            "API key is EMPTY for provider %s! The LLM request will likely fail with 401. "
            "Check keyring status and whether the key was saved correctly.", provider.id,
        )
    return OpenAICompatibleAdapter(
        base_url=provider.base_url,
        api_key=provider.api_key,
        model=provider.model,
        temperature=s.temperature,
        max_tokens=s.max_tokens,
        timeout=float(s.api_timeout),
    )


def create_app(db_path: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="SuperAgent", version="0.2.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    conn = db.connect(db_path)
    db.init_schema(conn)
    app.state.db = conn
    app_secrets.set_sqlite_conn(conn)
    app.state.adapter_factory = lambda provider_id=None: build_adapter(conn, provider_id)
    app.state.memory_worker = MemoryWorker(conn, app.state.adapter_factory)
    app.state.registry = ToolRegistry()
    app.state.registry.register(LoadSkillTool())
    app.state.registry.register(DispatchSubagentTool())
    tool_dirs = external_tools_dirs()
    ensure_tool_dirs(tool_dirs)
    app.state.loaded_tool_plugins = load_tool_plugins(app.state.registry, tool_dirs)

    builtin_skills = builtin_skills_dir()
    user_skill_dirs = [*external_skills_dirs(), default_user_skills_dir()]
    for user_skills in user_skill_dirs:
        Path(user_skills).mkdir(parents=True, exist_ok=True)
    ensure_external_persona_dirs()
    app.state.skills = SkillService(builtin_skills, user_skill_dirs)

    app.include_router(health.router)
    app.include_router(chat_ws.router)
    app.include_router(projects_api.router)
    app.include_router(sessions.router)
    app.include_router(settings_api.router)
    app.include_router(providers.router)
    app.include_router(memories.router)
    app.include_router(skills_api.router)
    app.include_router(tools_api.router)
    app.include_router(balance_router)

    # Serve the built frontend same-origin (packaged app). Mounted last so API
    # and WebSocket routes take precedence. Skipped if the frontend isn't built.
    dist = frontend_dist_dir()
    if os.path.isdir(dist):
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app


app = create_app()
