"""Workspace sandbox.

All file/command tools are confined to a workspace root. Paths that resolve
outside the root (via `..` or an absolute path) are rejected — this is a hard
safety boundary, not a convenience.
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path


class PathEscapeError(Exception):
    """Raised when a requested path resolves outside the workspace root."""


def ensure_workspace(root: str) -> str:
    p = Path(root).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def resolve_in_workspace(root: str, requested: str) -> Path:
    root_p = Path(root).expanduser().resolve()
    if os.path.isabs(requested):
        candidate = Path(requested).resolve()
    else:
        candidate = (root_p / requested).resolve()
    try:
        candidate.relative_to(root_p)
    except ValueError as exc:
        raise PathEscapeError(f"path escapes workspace: {requested}") from exc
    return candidate


def validate_command_paths(root: str, command: str) -> None:
    """Reject obvious path escapes before running a shell command.

    This is a guardrail, not a complete shell parser. It catches the cases that
    matter most for this app: absolute Windows paths and relative paths using
    `..` that would write/read outside the selected project.
    """
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', command)
    candidates = [a or b for a, b in quoted]
    try:
        candidates.extend(shlex.split(command, posix=False))
    except ValueError:
        candidates.extend(command.split())

    for raw in candidates:
        token = raw.strip().strip('"').strip("'").rstrip(";,")
        if not token:
            continue
        looks_like_path = os.path.isabs(token) or token.startswith("..") or "\\.." in token or "/.." in token
        if not looks_like_path:
            continue
        resolve_in_workspace(root, token)
