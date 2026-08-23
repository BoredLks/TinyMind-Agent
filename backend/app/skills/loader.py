"""Skill loading and metadata index.

A skill is a directory containing `SKILL.md` (YAML frontmatter with `name` and
`description`, plus a Markdown body). Skills load from a built-in (read-only,
shipped) directory and a user-writable directory; same-named user skills
override built-ins.

Progressive disclosure: only name+description are injected into the agent's
system prompt up front; the full body is fetched on demand (via the load_skill
tool) — mirroring the Superpowers approach.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml


@dataclass
class SkillMeta:
    name: str
    description: str
    source: str  # "builtin" | "user"
    path: str


def parse_frontmatter(text: str) -> Tuple[dict, str]:
    """Split `---`-delimited YAML frontmatter from the Markdown body."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    fm: List[str] = []
    i = 1
    while i < len(lines) and lines[i].strip() != "---":
        fm.append(lines[i])
        i += 1
    meta = yaml.safe_load("\n".join(fm)) or {}
    body = "\n".join(lines[i + 1:]).lstrip("\n")
    return (meta if isinstance(meta, dict) else {}), body


def default_user_skills_dir() -> str:
    override = os.getenv("SUPERAGENT_SKILLS_DIR")
    if override:
        return override
    base = os.getenv("APPDATA") or os.path.expanduser("~")
    return str(Path(base) / "SuperAgent" / "skills")


class SkillService:
    """Loads skill metadata + bodies from the built-in and user directories."""

    def __init__(self, builtin_dir: str, user_dir: str | Iterable[str]) -> None:
        self.builtin_dir = builtin_dir
        self.user_dirs = [user_dir] if isinstance(user_dir, str) else list(user_dir)
        self._metas: Dict[str, SkillMeta] = {}
        self._bodies: Dict[str, str] = {}
        self.reload()

    def reload(self) -> None:
        self._metas.clear()
        self._bodies.clear()
        self._scan(self.builtin_dir, "builtin")
        for user_dir in self.user_dirs:
            self._scan(user_dir, "user")  # user/ext dirs override built-in

    def _scan(self, root: str, source: str) -> None:
        if not root or not os.path.isdir(root):
            return
        for entry in sorted(os.listdir(root)):
            skill_md = os.path.join(root, entry, "SKILL.md")
            if not os.path.isfile(skill_md):
                continue
            try:
                text = Path(skill_md).read_text(encoding="utf-8")
            except OSError:
                continue
            meta, body = parse_frontmatter(text)
            name = str(meta.get("name") or entry).strip()
            desc = str(meta.get("description") or "").strip()
            self._metas[name] = SkillMeta(name=name, description=desc, source=source, path=skill_md)
            self._bodies[name] = body

    def names(self) -> List[str]:
        return list(self._metas.keys())

    def list_metas(self) -> List[SkillMeta]:
        return list(self._metas.values())

    def get_meta(self, name: str) -> Optional[SkillMeta]:
        return self._metas.get(name)

    def get_body(self, name: str) -> Optional[str]:
        return self._bodies.get(name)
