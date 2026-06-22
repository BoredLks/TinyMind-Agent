"""story 专家：编写小故事。

纯生成任务，契合 MiniMind 作为聊天模型的能力。流程：LLM 生成 → 落盘到 workspace/stories/。
模型输出过短/为空时给出兜底文本，保证一定有产物。
"""

from __future__ import annotations  # 启用延迟注解求值

import os                            # 拆分路径、判断文件是否存在
import re                            # 正则：从指令提炼标题、清理文件名
from typing import Any               # 类型标注

from langchain_core.language_models.chat_models import BaseChatModel  # 模型类型
from langchain_core.messages import HumanMessage, SystemMessage       # 构造对话消息

from minimind_agent.prompts import STORY_SYSTEM, STORY_USER  # 写故事的提示词
from minimind_agent.workspace import Workspace               # 沙箱（落盘）


def _title_from_instruction(instruction: str) -> str:  # 从指令里提炼一个简短标题
    """从指令里取一个简短标题（去掉“写一个/关于”等套话）。"""
    text = instruction.strip()       # 去首尾空白
    text = re.sub(r"^(请|帮我|麻烦|给我)?\s*(写|编|创作|来)(一篇|一个|个)?", "", text)  # 去开头“请写一个”等套话
    text = re.sub(r"(的)?(小?故事|童话|小说|作文)$", "", text).strip()  # 去结尾“的小故事/童话”等
    text = text.strip("，,。.：: ")   # 去残留标点
    return (text or "小故事")[:20]   # 兜底“小故事”，并截断到 20 字


def _slug(title: str) -> str:        # 把标题转成安全的文件名片段
    """把标题转成相对安全的文件名片段（中文直接保留，去掉非法字符）。"""
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", title).strip("_")  # 非法字符/空白替换为下划线
    return cleaned or "story"        # 全被替换掉则用 story


def _unique_rel_path(workspace: Workspace, rel: str) -> str:  # 目标已存在则加序号（同 code_agent）
    if not (workspace.root / rel).exists():  # 不存在直接用
        return rel
    head, tail = os.path.split(rel)  # 拆目录 + 文件名
    stem, ext = os.path.splitext(tail)  # 拆主名 + 扩展名
    index = 2
    while True:
        candidate = f"{head}/{stem}_{index}{ext}" if head else f"{stem}_{index}{ext}"  # 候选名
        if not (workspace.root / candidate).exists():
            return candidate
        index += 1


def _fallback_story(instruction: str) -> str:  # 模型输出过短时的占位故事
    theme = _title_from_instruction(instruction)  # 提炼主题
    return (
        f"关于「{theme}」的小故事：\n\n"
        "（模型本次没有生成足够内容，这是一段占位故事。）\n"
        f"很久很久以前，有一个关于{theme}的小小约定，它在时间里慢慢长大，"
        "最终变成了一个温暖的结局。"
    )


def run_story_agent(                 # 执行一个“写小故事”步骤，返回结构化结果
    model: BaseChatModel,
    workspace: Workspace,
    step: dict[str, Any],
) -> dict[str, Any]:
    """执行一个“写小故事”步骤，返回结构化结果。"""
    instruction = str(step.get("instruction", ""))  # 该步骤指令
    try:
        response = model.invoke(     # 让模型写故事
            [
                SystemMessage(content=STORY_SYSTEM),
                HumanMessage(content=STORY_USER.format(instruction=instruction)),
            ]
        )
        story = str(getattr(response, "content", "")).strip()  # 取故事正文
    except Exception as exc:         # 模型不可用
        story = ""
        model_error = f"{type(exc).__name__}: {exc}"
    else:
        model_error = ""

    used_fallback = False            # 是否用了占位兜底
    if len(story) < 15:  # 过短视为无效输出
        story = _fallback_story(instruction)
        used_fallback = True

    title = _title_from_instruction(instruction)  # 标题
    document = f"# {title}\n\n{story}\n"           # 组成 Markdown（标题 + 正文）
    rel = _unique_rel_path(workspace, f"stories/{_slug(title)}.md")  # 落盘相对路径（不覆盖）
    write_result = workspace.write_text(rel, document)  # 写文件
    if not write_result.get("ok"):   # 写失败
        return {"ok": False, "result": write_result.get("error", "写文件失败"), "artifact": "", "error": "write_failed"}

    artifact = write_result.get("path", rel)  # 产物路径
    note = "（模型输出过短，已用占位故事" + (f"：{model_error}" if model_error else "") + "）" if used_fallback else ""  # 兜底说明
    return {
        "ok": True,
        "result": f"已生成故事 {artifact}（约 {len(story)} 字）{note}",
        "artifact": artifact,
        "used_fallback": used_fallback,  # True=故事是Python占位兜底；False=故事来自模型
    }
