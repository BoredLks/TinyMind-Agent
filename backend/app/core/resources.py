"""Resource path resolution for dev, PyInstaller onefile, and onedir builds.

Bundled read-only data lives under sys._MEIPASS when frozen. User-extensible
data lives beside the executable in onedir builds (or under the repo root in
dev) and under %APPDATA% as a per-user fallback.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _meipass():
    return getattr(sys, "_MEIPASS", None)


def app_dir() -> Path:
    """Directory beside the executable (frozen) or repo root (dev)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[3]


def builtin_skills_dir() -> str:
    """Directory of shipped (read-only) skills.

    In dev mode, returns backend/app/skills_builtin.
    In packaged mode, skills are copied to dist/SuperAgent/skills
    (user-writable) instead of bundled inside _internal.
    Returns an empty string if the directory doesn't exist.
    """
    base = _meipass()
    if base:
        # Packaged mode: skills_builtin is no longer bundled; skills live in
        # dist/SuperAgent/skills (returned by external_skills_dirs).
        bundled = os.path.join(base, "skills_builtin")
        if os.path.isdir(bundled):
            return bundled
        return ""
    # dev: backend/app/skills_builtin  (resources.py -> core -> app)
    dev_dir = str(Path(__file__).resolve().parents[1] / "skills_builtin")
    return dev_dir if os.path.isdir(dev_dir) else ""


def builtin_personas_dir() -> str:
    """Directory of shipped (read-only) prompt/persona fragments."""
    base = _meipass()
    if base:
        return os.path.join(base, "personas_builtin")
    return str(Path(__file__).resolve().parents[1] / "personas_builtin")


def external_skills_dirs() -> list[str]:
    """Writable skill directories, checked after built-ins so users can extend/override."""
    dirs = [app_dir() / "skills"]
    appdata = os.getenv("APPDATA")
    if appdata:
        dirs.append(Path(appdata) / "SuperAgent" / "skills")
    return [str(p) for p in dirs]


def external_persona_dirs() -> list[str]:
    """Writable prompt/persona fragment directories."""
    dirs = [app_dir() / "personas"]
    appdata = os.getenv("APPDATA")
    if appdata:
        dirs.append(Path(appdata) / "SuperAgent" / "personas")
    return [str(p) for p in dirs]


def external_tools_dirs() -> list[str]:
    """Python tool plugin directories."""
    dirs = [app_dir() / "tools"]
    appdata = os.getenv("APPDATA")
    if appdata:
        dirs.append(Path(appdata) / "SuperAgent" / "tools")
    return [str(p) for p in dirs]


def frontend_dist_dir() -> str:
    """Directory of the built frontend (served same-origin by FastAPI)."""
    base = _meipass()
    if base:
        return os.path.join(base, "frontend_dist")
    # dev: <repo>/frontend/dist  (resources.py -> core -> app -> backend -> repo)
    return str(Path(__file__).resolve().parents[3] / "frontend" / "dist")
