"""Data access for sessions, messages, and settings.

Functions take an open sqlite3.Connection so they are trivially testable
against an in-memory database. Rows are returned as plain dicts.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


# --- sessions -------------------------------------------------------------

def create_session(
    conn: sqlite3.Connection,
    title: str = "新会话",
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    meta_json: Any = None,
    project_id: Optional[str] = None,
) -> dict:
    sid = _new_id()
    now = _now()
    meta_value = json.dumps(meta_json, ensure_ascii=False) if meta_json is not None else None
    conn.execute(
        "INSERT INTO sessions (id, title, created_at, updated_at, model, system_prompt, meta_json, project_id)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (sid, title, now, now, model, system_prompt, meta_value, project_id),
    )
    conn.commit()
    return get_session(conn, sid)  # type: ignore[return-value]


def set_session_project(
    conn: sqlite3.Connection, session_id: str, project_id: Optional[str]
) -> Optional[dict]:
    conn.execute(
        "UPDATE sessions SET project_id = ?, updated_at = ? WHERE id = ?",
        (project_id, _now(), session_id),
    )
    conn.commit()
    return get_session(conn, session_id)


def get_session(conn: sqlite3.Connection, session_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return dict(row) if row else None


def set_session_meta(conn: sqlite3.Connection, session_id: str, meta_json: Any) -> Optional[dict]:
    conn.execute(
        "UPDATE sessions SET meta_json = ?, updated_at = ? WHERE id = ?",
        (json.dumps(meta_json, ensure_ascii=False), _now(), session_id),
    )
    conn.commit()
    return get_session(conn, session_id)


def list_sessions(conn: sqlite3.Connection) -> List[dict]:
    rows = conn.execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall()
    return [dict(r) for r in rows]


def rename_session(conn: sqlite3.Connection, session_id: str, title: str) -> Optional[dict]:
    conn.execute(
        "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
        (title, _now(), session_id),
    )
    conn.commit()
    return get_session(conn, session_id)


def touch_session(conn: sqlite3.Connection, session_id: str) -> None:
    conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (_now(), session_id))
    conn.commit()


def set_workflow_stage(conn: sqlite3.Connection, session_id: str, stage: str) -> None:
    conn.execute("UPDATE sessions SET workflow_stage = ? WHERE id = ?", (stage, session_id))
    conn.commit()


def add_session_tokens(conn: sqlite3.Connection, session_id: str, tokens: int) -> int:
    conn.execute(
        "UPDATE sessions SET token_total = token_total + ? WHERE id = ?", (tokens, session_id)
    )
    conn.commit()
    row = conn.execute("SELECT token_total FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return int(row["token_total"]) if row else tokens


def delete_session(conn: sqlite3.Connection, session_id: str) -> None:
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()


# --- projects -------------------------------------------------------------

def create_project(conn: sqlite3.Connection, name: str, path: str) -> dict:
    existing = get_project_by_path(conn, path)
    if existing:
        rename_project(conn, existing["id"], name or existing["name"])
        return get_project(conn, existing["id"])  # type: ignore[return-value]

    pid = _new_id()
    now = _now()
    conn.execute(
        "INSERT INTO projects (id, name, path, created_at, updated_at, last_opened_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (pid, name, path, now, now, now),
    )
    conn.commit()
    return get_project(conn, pid)  # type: ignore[return-value]


def get_project(conn: sqlite3.Connection, project_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    return dict(row) if row else None


def get_project_by_path(conn: sqlite3.Connection, path: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM projects WHERE path = ?", (path,)).fetchone()
    return dict(row) if row else None


def list_projects(conn: sqlite3.Connection) -> List[dict]:
    rows = conn.execute(
        "SELECT * FROM projects ORDER BY COALESCE(last_opened_at, updated_at) DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def rename_project(conn: sqlite3.Connection, project_id: str, name: str) -> Optional[dict]:
    conn.execute(
        "UPDATE projects SET name = ?, updated_at = ? WHERE id = ?",
        (name, _now(), project_id),
    )
    conn.commit()
    return get_project(conn, project_id)


def touch_project(conn: sqlite3.Connection, project_id: str) -> Optional[dict]:
    now = _now()
    conn.execute(
        "UPDATE projects SET last_opened_at = ?, updated_at = ? WHERE id = ?",
        (now, now, project_id),
    )
    conn.commit()
    return get_project(conn, project_id)


def delete_project(conn: sqlite3.Connection, project_id: str) -> None:
    # Detach the project's sessions instead of deleting them — they fall back
    # to the default sandbox workspace and move to the ungrouped list.
    conn.execute("UPDATE sessions SET project_id = NULL WHERE project_id = ?", (project_id,))
    conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    conn.commit()


def project_path_for_session(conn: sqlite3.Connection, session_id: str) -> Optional[str]:
    """Return the project directory a session is bound to, or None for sandbox."""
    row = conn.execute(
        "SELECT p.path AS path FROM sessions s"
        " JOIN projects p ON p.id = s.project_id WHERE s.id = ?",
        (session_id,),
    ).fetchone()
    return row["path"] if row else None


# --- messages -------------------------------------------------------------

def add_message(
    conn: sqlite3.Connection,
    session_id: str,
    role: str,
    content: str,
    token_in: Optional[int] = None,
    token_out: Optional[int] = None,
) -> dict:
    mid = _new_id()
    now = _now()
    conn.execute(
        "INSERT INTO messages (id, session_id, role, content, created_at, token_in, token_out, meta_json)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (mid, session_id, role, content, now, token_in, token_out, None),
    )
    conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
    conn.commit()
    row = conn.execute("SELECT * FROM messages WHERE id = ?", (mid,)).fetchone()
    return dict(row)


def list_messages(conn: sqlite3.Connection, session_id: str) -> List[dict]:
    rows = conn.execute(
        "SELECT * FROM messages WHERE session_id = ? ORDER BY rowid",
        (session_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def delete_last_turn(conn: sqlite3.Connection, session_id: str, rounds: int = 1) -> int:
    """Delete the last N user-started turns from a session."""
    rounds = max(1, rounds)
    rows = conn.execute(
        "SELECT rowid, role FROM messages WHERE session_id = ? ORDER BY rowid",
        (session_id,),
    ).fetchall()
    if not rows:
        return 0
    user_rows = [r["rowid"] for r in rows if r["role"] == "user"]
    if not user_rows:
        return 0
    cutoff = user_rows[-rounds] if len(user_rows) >= rounds else user_rows[0]
    cur = conn.execute(
        "DELETE FROM messages WHERE session_id = ? AND rowid >= ?",
        (session_id, cutoff),
    )
    touch_session(conn, session_id)
    return cur.rowcount


# --- long-term memories ---------------------------------------------------


def add_memory(conn: sqlite3.Connection, theme: str, content: str) -> dict:
    mid = _new_id()
    now = _now()
    conn.execute(
        "INSERT INTO memories (id, theme, content, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (mid, theme, content, now, now),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM memories WHERE id = ?", (mid,)).fetchone()
    return dict(row)


def list_memories(conn: sqlite3.Connection) -> List[dict]:
    rows = conn.execute("SELECT * FROM memories ORDER BY updated_at DESC").fetchall()
    return [dict(r) for r in rows]


def get_memory(conn: sqlite3.Connection, memory_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    return dict(row) if row else None


def update_memory(
    conn: sqlite3.Connection,
    memory_id: str,
    *,
    theme: Optional[str] = None,
    content: Optional[str] = None,
) -> Optional[dict]:
    existing = get_memory(conn, memory_id)
    if existing is None:
        return None
    conn.execute(
        "UPDATE memories SET theme = ?, content = ?, updated_at = ? WHERE id = ?",
        (
            theme if theme is not None else existing["theme"],
            content if content is not None else existing["content"],
            _now(),
            memory_id,
        ),
    )
    conn.commit()
    return get_memory(conn, memory_id)


def merge_memories(
    conn: sqlite3.Connection,
    keep_id: str,
    remove_id: str,
    *,
    theme: str,
    content: str,
) -> bool:
    if keep_id == remove_id:
        return False
    if get_memory(conn, keep_id) is None or get_memory(conn, remove_id) is None:
        return False
    update_memory(conn, keep_id, theme=theme, content=content)
    delete_memory(conn, remove_id)
    return True


def delete_memory(conn: sqlite3.Connection, memory_id: str) -> bool:
    cur = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    conn.commit()
    return cur.rowcount > 0


def clear_memories(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM memories")
    conn.commit()


# --- settings -------------------------------------------------------------

def get_setting(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = conn.execute("SELECT value_json FROM settings WHERE key = ?", (key,)).fetchone()
    return json.loads(row["value_json"]) if row else default


def set_setting(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        "INSERT INTO settings (key, value_json) VALUES (?, ?)"
        " ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
        (key, json.dumps(value)),
    )
    conn.commit()


def get_all_settings(conn: sqlite3.Connection) -> dict:
    rows = conn.execute("SELECT key, value_json FROM settings").fetchall()
    return {r["key"]: json.loads(r["value_json"]) for r in rows}


# --- skill state ----------------------------------------------------------

def is_skill_enabled(conn: sqlite3.Connection, name: str, default: bool = True) -> bool:
    row = conn.execute("SELECT enabled FROM skill_state WHERE name = ?", (name,)).fetchone()
    return bool(row["enabled"]) if row else default


def set_skill_enabled(conn: sqlite3.Connection, name: str, enabled: bool) -> None:
    conn.execute(
        "INSERT INTO skill_state (name, enabled) VALUES (?, ?)"
        " ON CONFLICT(name) DO UPDATE SET enabled = excluded.enabled",
        (name, 1 if enabled else 0),
    )
    conn.commit()


def get_skill_states(conn: sqlite3.Connection) -> dict:
    rows = conn.execute("SELECT name, enabled FROM skill_state").fetchall()
    return {r["name"]: bool(r["enabled"]) for r in rows}
