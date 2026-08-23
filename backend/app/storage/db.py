"""SQLite persistence: connection management and schema.

M1 kept conversation state in memory. M2 persists sessions, messages, and
non-secret settings here. Secrets (API keys) never live in this DB — they go
to the OS keyring (see core/secrets.py).

The DB lives under %APPDATA%/SuperAgent so user data stays out of the repo.
Tests pass ":memory:" for an isolated in-process database.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    model          TEXT,
    system_prompt  TEXT,
    meta_json      TEXT,
    workflow_stage TEXT NOT NULL DEFAULT 'idle',
    token_total    INTEGER NOT NULL DEFAULT 0,
    project_id     TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    token_in    INTEGER,
    token_out   INTEGER,
    meta_json   TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    path           TEXT NOT NULL UNIQUE,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    last_opened_at TEXT
);

CREATE TABLE IF NOT EXISTS skill_state (
    name    TEXT PRIMARY KEY,
    enabled INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    theme       TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""


def default_db_path() -> str:
    """Per-user database path. Overridable via SUPERAGENT_DB_PATH (tests use this)."""
    override = os.getenv("SUPERAGENT_DB_PATH")
    if override:
        return override
    base = os.getenv("APPDATA") or str(Path.home())
    directory = Path(base) / "SuperAgent"
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory / "superagent.db")


def connect(path: Optional[str] = None) -> sqlite3.Connection:
    """Open a connection with row access by name and FK enforcement on.

    check_same_thread=False because FastAPI may dispatch DB work across the
    asyncio threadpool; access is serialized at the call sites in M2.
    """
    if path is None:
        path = default_db_path()
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    _migrate(conn)
    conn.commit()


def _migrate(conn: sqlite3.Connection) -> None:
    """Lightweight additive migrations for already-existing databases."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()}
    if "workflow_stage" not in cols:
        conn.execute(
            "ALTER TABLE sessions ADD COLUMN workflow_stage TEXT NOT NULL DEFAULT 'idle'"
        )
    if "token_total" not in cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN token_total INTEGER NOT NULL DEFAULT 0")
    if "project_id" not in cols:
        # Associate a session with a project directory; NULL = default sandbox.
        conn.execute("ALTER TABLE sessions ADD COLUMN project_id TEXT")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id             TEXT PRIMARY KEY,
            name           TEXT NOT NULL,
            path           TEXT NOT NULL UNIQUE,
            created_at     TEXT NOT NULL,
            updated_at     TEXT NOT NULL,
            last_opened_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id          TEXT PRIMARY KEY,
            theme       TEXT NOT NULL,
            content     TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )
        """
    )
