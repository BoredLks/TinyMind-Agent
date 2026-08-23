"""Workflow stage tracking.

A session moves through brainstorm -> plan -> execute -> review. Stages advance
automatically when the agent loads the corresponding skill; the UI shows the
current stage. (Explicit transitions can be layered on later.)
"""

from __future__ import annotations

from typing import Optional

STAGES = ["brainstorm", "plan", "execute", "review"]

_STAGE_FOR_SKILL = {
    "brainstorming": "brainstorm",
    "writing-plans": "plan",
    "test-driven-development": "execute",
}


def stage_for_skill(skill_name: Optional[str]) -> Optional[str]:
    if not skill_name:
        return None
    return _STAGE_FOR_SKILL.get(skill_name)
