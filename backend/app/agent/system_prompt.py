"""System-prompt assembly: persona + skill index (progressive disclosure) + hints.

Only skill name+description go into the prompt up front; the agent fetches a
skill's full body on demand via the load_skill tool. A lightweight rule-based
hint nudges the agent toward an obviously-relevant skill (the "rules" half of
auto-triggering; the LLM's own judgment is the other half).
"""

from __future__ import annotations

from typing import List, Optional

from app.agent.personas import build_persona_prompt

PERSONA = (
    "你是 SuperAgent，一个遵循工程方法论的编码/开发助手。\n"
    "你拥有一组「技能」（经过验证的方法论指南）。在动手做任何事之前，如果某个技能可能适用"
    "（哪怕只有 1% 相关），先调用 load_skill 工具加载它，并严格遵循其内容。\n"
    "流程类技能（如 brainstorming）优先于实现类技能。用户的显式指令高于技能。\n"
    "工具采用两步调用策略：系统只给出工具概览；首次使用复杂工具、工具说明要求先读文档、"
    "或工具调用失败后，需要先调用 load_tool_doc 获取该工具的完整说明。\n"
    "当继续执行必须依赖用户选择、账号、路径意图或安全确认时，使用 ask_user_qa / "
    "ask_user_single_choice / ask_user_multi_choice 暂停等待用户输入。"
)

_BUILD_WORDS = (
    "做一个", "做个", "构建", "搭建", "实现", "写一个", "新功能", "做一款",
    "build", "create", "feature", "implement",
)


def build_system_prompt(
    enabled_skills: List,  # List[SkillMeta]
    user_prompt: Optional[str] = None,
    hint: Optional[str] = None,
    workspace_root: Optional[str] = None,
    memory: Optional[str] = None,
) -> str:
    parts = [build_persona_prompt(fallback=PERSONA)]
    if workspace_root:
        parts.append(
            "\n当前项目目录 / 工具工作区：\n"
            f"{workspace_root}\n"
            "所有文件读写、目录创建、命令执行都必须限制在这个项目目录内。"
            "不要尝试在项目目录之外创建或移动文件；需要跨目录操作时先向用户说明限制。"
            "调用 run_command 时默认使用 Windows PowerShell 语法，并优先使用相对路径；"
            "只有遇到 cmd.exe 专用语法时，才把 shell 参数设为 cmd。"
        )
    if enabled_skills:
        lines = ["", "可用技能（用 load_skill 加载全文）："]
        for m in enabled_skills:
            lines.append(f"- {m.name}：{m.description}")
        parts.append("\n".join(lines))
    if hint:
        parts.append("\n" + hint)
    if memory:
        parts.append("\n用户长期记忆（仅作背景，若与用户当前说法冲突，以当前说法为准）：\n" + memory)
    if user_prompt:
        parts.append("\n用户自定义指令：\n" + user_prompt)
    return "\n".join(parts)


def suggest_skill(user_text: str, available: List[str]) -> Optional[str]:
    """Rule-based nudge toward a relevant skill, or None."""
    lowered = user_text.lower()
    if "brainstorming" in available and any(
        w in user_text or w in lowered for w in _BUILD_WORDS
    ):
        return "（提示：这看起来像要创建/构建东西——建议先 load_skill 加载 brainstorming，先澄清需求与设计再动手。）"
    return None
