"""Asynchronous long-term memory maintenance.

The main chat turn enqueues finished turns and never waits for memory work.
The background worker asks the configured model for a small CRUD plan, then
applies create/update/delete/merge operations to SQLite.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from typing import Callable, Iterable

from app.storage import dao


def format_memory_narrative(conn: sqlite3.Connection) -> str:
    rows = dao.list_memories(conn)
    if not rows:
        return ""
    grouped: dict[str, list[str]] = {}
    for row in rows:
        grouped.setdefault(row["theme"], []).append(row["content"])
    lines = ["## 长期记忆"]
    for theme, items in grouped.items():
        lines.append(f"### {theme}")
        for item in items[:20]:
            lines.append(f"- {item}")
    return "\n".join(lines)


class MemoryWorker:
    def __init__(self, conn: sqlite3.Connection, adapter_factory: Callable) -> None:
        self._conn = conn
        self._adapter_factory = adapter_factory
        self._queue: asyncio.Queue[dict | None] | None = None
        self._task: asyncio.Task | None = None

    def enqueue(self, *, user: str, assistant: str) -> None:
        if not user.strip() or not assistant.strip():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._queue is None:
            self._queue = asyncio.Queue()
        if self._task is None or self._task.done():
            self._task = loop.create_task(self._run())
        self._queue.put_nowait({"user": user, "assistant": assistant})

    async def _run(self) -> None:
        assert self._queue is not None
        while True:
            item = await self._queue.get()
            if item is None:
                return
            try:
                await self._consume(item["user"], item["assistant"])
            except Exception:
                # Memory must never break the chat turn.
                pass

    async def _consume(self, user: str, assistant: str) -> None:
        existing_rows = dao.list_memories(self._conn)
        prompt = _build_memory_prompt(existing_rows, user, assistant)
        adapter = self._make_adapter()
        text_parts: list[str] = []
        async for kind, payload in adapter.stream_turn([{"role": "user", "content": prompt}], None):
            if kind == "text":
                text_parts.append(payload)
        payload = _parse_json("".join(text_parts))
        operations = _normalise_operations(payload)
        _apply_operations(self._conn, operations)

    def _make_adapter(self):
        try:
            return self._adapter_factory()
        except TypeError:
            return self._adapter_factory(None)


def _build_memory_prompt(existing_rows: list[dict], user: str, assistant: str) -> str:
    existing = _format_existing_for_prompt(existing_rows)
    return (
        "You are a background long-term memory maintainer for a chat agent.\n"
        "Extract only durable facts explicitly stated by the user. Do not store ordinary greetings, "
        "temporary task details, tool outputs, or guesses.\n"
        "Compare the new turn with existing memories and return only JSON.\n\n"
        "Allowed operations:\n"
        "- create: add a new durable user fact.\n"
        "- update: replace an existing memory when the user corrected or refined it.\n"
        "- delete: remove an obsolete or contradicted memory.\n"
        "- merge: combine two duplicate/overlapping memories; keep keep_id and remove remove_id.\n\n"
        "Return schema:\n"
        "{\"operations\":["
        "{\"op\":\"create\",\"theme\":\"1-4 word category\",\"content\":\"third-person concise fact\"},"
        "{\"op\":\"update\",\"id\":\"existing id\",\"theme\":\"category\",\"content\":\"replacement fact\"},"
        "{\"op\":\"delete\",\"id\":\"existing id\",\"reason\":\"why\"},"
        "{\"op\":\"merge\",\"keep_id\":\"id\",\"remove_id\":\"id\",\"theme\":\"category\",\"content\":\"merged fact\"}"
        "]}\n"
        "If nothing should be remembered, return {\"operations\":[]}.\n\n"
        f"Existing memories:\n{existing}\n\n"
        f"User turn:\n{user}\n\n"
        f"Assistant turn:\n{assistant}"
    )


def _format_existing_for_prompt(rows: Iterable[dict]) -> str:
    items = list(rows)
    if not items:
        return "(none)"
    lines = []
    for row in items:
        lines.append(f"- id={row['id']} theme={row['theme']} content={row['content']}")
    return "\n".join(lines)


def _normalise_operations(payload: dict) -> list[dict]:
    raw_ops = payload.get("operations")
    if isinstance(raw_ops, list):
        return [op for op in raw_ops if isinstance(op, dict)]

    # Backward-compatible with the previous extractor-only JSON shape.
    memories = payload.get("memories")
    if isinstance(memories, list):
        ops: list[dict] = []
        for entry in memories:
            if not isinstance(entry, dict):
                continue
            ops.append(
                {
                    "op": "create",
                    "theme": entry.get("theme"),
                    "content": entry.get("content"),
                }
            )
        return ops
    return []


def _apply_operations(conn: sqlite3.Connection, operations: Iterable[dict]) -> None:
    for operation in operations:
        op = str(operation.get("op") or "").strip().lower()
        if op == "create":
            _apply_create(conn, operation)
        elif op == "update":
            _apply_update(conn, operation)
        elif op == "delete":
            memory_id = str(operation.get("id") or "").strip()
            if memory_id:
                dao.delete_memory(conn, memory_id)
        elif op == "merge":
            _apply_merge(conn, operation)


def _apply_create(conn: sqlite3.Connection, operation: dict) -> None:
    theme = _clean_theme(operation.get("theme"))
    content = _clean_content(operation.get("content"))
    if not content or _memory_exists(conn, theme, content):
        return
    dao.add_memory(conn, theme, content)


def _apply_update(conn: sqlite3.Connection, operation: dict) -> None:
    memory_id = str(operation.get("id") or "").strip()
    content = _clean_content(operation.get("content"))
    if not memory_id or not content:
        return
    dao.update_memory(conn, memory_id, theme=_clean_theme(operation.get("theme")), content=content)


def _apply_merge(conn: sqlite3.Connection, operation: dict) -> None:
    keep_id = str(operation.get("keep_id") or "").strip()
    remove_id = str(operation.get("remove_id") or "").strip()
    content = _clean_content(operation.get("content"))
    if not keep_id or not remove_id or not content:
        return
    dao.merge_memories(
        conn,
        keep_id,
        remove_id,
        theme=_clean_theme(operation.get("theme")),
        content=content,
    )


def _memory_exists(conn: sqlite3.Connection, theme: str, content: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM memories WHERE theme = ? AND content = ? LIMIT 1",
        (theme, content),
    ).fetchone()
    return row is not None


def _clean_theme(value) -> str:
    text = str(value or "偏好").strip()
    return text[:24] or "偏好"


def _clean_content(value) -> str:
    text = str(value or "").strip()
    return text.replace("\r", " ").replace("\n", " ")[:500]


def _parse_json(text: str) -> dict:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {}
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}

