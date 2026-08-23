"""Prompt fragment loading for persona/system instructions.

The built-in prompt is shipped read-only, while user fragments can live beside
the executable or in %APPDATA%. Later directories override files with the same
name, and extra Markdown files are appended in filename order.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from app.core.resources import builtin_personas_dir, external_persona_dirs

_PRIORITY = {
    "AGENTS.md": 10,
    "STYLE.md": 20,
    "USER.md": 30,
    "MEMORY.md": 40,
}


@dataclass(frozen=True)
class PromptFragment:
    name: str
    content: str
    source: str
    path: str


def load_prompt_fragments(
    builtin_dir: str | None = None,
    user_dirs: Iterable[str] | None = None,
) -> List[PromptFragment]:
    """Load Markdown prompt fragments from built-in and user directories."""
    roots: list[tuple[str, str]] = []
    builtin = builtin_dir if builtin_dir is not None else builtin_personas_dir()
    if builtin:
        roots.append(("builtin", builtin))
    for directory in user_dirs if user_dirs is not None else external_persona_dirs():
        if directory:
            roots.append(("user", directory))

    by_name: dict[str, PromptFragment] = {}
    for source, root in roots:
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        for path in sorted(root_path.glob("*.md")):
            if path.name.startswith(".") or path.name.endswith(".example.md"):
                continue
            try:
                content = path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if not content:
                continue
            by_name[path.name] = PromptFragment(
                name=path.name,
                content=content,
                source=source,
                path=str(path),
            )

    return sorted(
        by_name.values(),
        key=lambda item: (_PRIORITY.get(item.name, 100), item.name.lower()),
    )


def build_persona_prompt(
    fragments: Iterable[PromptFragment] | None = None,
    *,
    fallback: str = "",
) -> str:
    """Assemble prompt fragments, falling back to the legacy in-code prompt."""
    loaded = list(fragments) if fragments is not None else load_prompt_fragments()
    if not loaded:
        return fallback
    sections = []
    for fragment in loaded:
        title = Path(fragment.name).stem
        sections.append(f"## {title}\n{fragment.content}")
    return "\n\n".join(sections)


def ensure_external_persona_dirs() -> None:
    """Create writable persona directories and a small README for users."""
    for directory in external_persona_dirs():
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        readme = path / "README.md"
        if readme.exists():
            continue
        readme.write_text(
            "# SuperAgent persona fragments\n\n"
            "Put trusted Markdown prompt fragments here to extend or override the built-in system prompt.\n\n"
            "Common filenames are `AGENTS.md`, `STYLE.md`, `USER.md`, and `MEMORY.md`.\n"
            "Files with the same name override built-in fragments; other `.md` files are appended.\n",
            encoding="utf-8",
        )

