"""file 专家：移动文件。

设计取舍：MiniMind 这类超小模型对“结构化抽取”不可靠，因此“源/目标”优先用**确定性正则**
从指令里抽取；正则失败时才退回让模型抽取 JSON。真正的移动动作由 workspace 沙箱
（``shutil.move`` + 越界校验）确定性完成——这部分绝不依赖模型，保证演示稳定可复现。
"""

from __future__ import annotations  # 启用延迟注解求值

import re                            # 正则：切子句、抽取“源→目标”
from typing import Any, Optional     # 类型标注

from langchain_core.language_models.chat_models import BaseChatModel  # 模型类型（仅兜底抽取时用）
from langchain_core.messages import HumanMessage, SystemMessage       # 兜底抽取构造消息用

from minimind_agent.parsing import extract_json  # 从模型回复里容错解析 JSON
from minimind_agent.prompts import FILE_EXTRACT_SYSTEM, FILE_EXTRACT_USER  # 兜底抽取提示词
from minimind_agent.workspace import Workspace   # 沙箱（真正执行移动）


# 把指令切成子句时用到的“强分隔符”（顺序连接词 + 标点）
_CLAUSE_SPLIT = re.compile(r"然后|接着|之后|再|并且|并|最后|最终|[，,。；;、\n]")

# “把/将 X 移动到 Y”一类的抽取模式（英文用 IGNORECASE）
_MOVE_PATTERNS = [
    r"把\s*(.+?)\s*(?:移动|移|挪)\s*(?:到|至|进)\s*(.+)",   # 把 X 移动到 Y
    r"将\s*(.+?)\s*(?:移动|移|挪)\s*(?:到|至|进)\s*(.+)",   # 将 X 移到 Y
    r"把\s*(.+?)\s*放\s*(?:到|进|入)\s*(.+)",               # 把 X 放到 Y
    r"把\s*(.+?)\s*重命名为\s*(.+)",                         # 把 X 重命名为 Y
    r"将\s*(.+?)\s*重命名为\s*(.+)",                         # 将 X 重命名为 Y
    r"(.+?)\s*(?:移动|移|挪)\s*(?:到|至|进)\s*(.+)",        # X 移动到 Y（无“把/将”）
    r"move\s+(.+?)\s+(?:to|into)\s+(.+)",                     # move X to Y
    r"mv\s+(\S+)\s+(\S+)",                                    # mv X Y
]

# 目标若以这些词结尾，说明它是“目录”
_DIR_WORDS = ("目录", "文件夹", "文件目录")          # 强目录词
_DIR_TAIL_WORDS = ("下", "里", "中", "内")          # 方位词（也暗示目录）


def _clean_token(token: str) -> str:  # 清理抽取到的路径 token
    """清理抽取到的路径 token：去引号、去首尾标点空白。"""
    token = token.strip()            # 去首尾空白
    token = token.strip("「」『』《》“”‘’\"' ")  # 去各种引号
    token = token.strip("。.,，；;:：!！?？、 ")   # 去首尾标点
    return token.strip()


def _focus_clause(instruction: str) -> str:  # 从多子句指令里挑出“含移动关键词”的那段
    """从（可能含多个子句的）指令里，取出含移动关键词的那一段。"""
    pieces = [p.strip() for p in _CLAUSE_SPLIT.split(instruction) if p.strip()]  # 切成子句并去空
    for piece in pieces:             # 找第一个含移动类关键词的子句
        if any(k in piece for k in ("移动", "移到", "移至", "挪", "move", "mv ", "重命名", "放到")):
            return piece
    return instruction               # 没找到就返回整条指令


def _regex_extract(instruction: str) -> Optional[tuple[str, str, bool]]:  # 正则抽取 (源, 目标, 目标是否目录)
    """正则抽取 (source, destination, dest_is_dir)，失败返回 None。"""
    for pattern in _MOVE_PATTERNS:   # 逐个模式尝试
        match = re.search(pattern, instruction, flags=re.IGNORECASE)
        if not match:
            continue                 # 该模式没匹配上，试下一个
        src_raw, dst_raw = match.group(1), match.group(2)  # 捕获的源/目标原始片段
        # 源：取最后一个子句片段（避免把前置子句吃进来）
        src = _clean_token(re.split(_CLAUSE_SPLIT, src_raw)[-1])
        # 目标：取第一个子句片段（避免把后置子句吃进来）
        dst = _clean_token(re.split(_CLAUSE_SPLIT, dst_raw)[0])
        if not src or not dst:       # 清理后为空则放弃这次匹配
            continue
        dest_is_dir = False          # 推断目标是不是目录
        for word in _DIR_WORDS:      # 以“目录/文件夹”等结尾 → 是目录，并去掉该词
            if dst.endswith(word):
                dst = _clean_token(dst[: -len(word)].rstrip("的"))
                dest_is_dir = True
        for word in _DIR_TAIL_WORDS:  # 以“下/里/中/内”结尾 → 也视为目录
            if dst.endswith(word) and len(dst) > 1:
                dst = _clean_token(dst[:-1])
                dest_is_dir = True
        if src and dst:              # 成功抽到非空源/目标
            return src, dst, dest_is_dir
    return None                      # 所有模式都失败


def _llm_extract(model: BaseChatModel, instruction: str) -> Optional[tuple[str, str, bool]]:  # 正则失败时让模型抽
    """正则失败时，让模型抽取 JSON {source, destination}。"""
    try:
        response = model.invoke(     # 让模型按 JSON 抽取源/目标
            [
                SystemMessage(content=FILE_EXTRACT_SYSTEM),
                HumanMessage(content=FILE_EXTRACT_USER.format(instruction=instruction)),
            ]
        )
    except Exception:                # 模型不可用（如 server 未启动）则放弃兜底
        return None
    data = extract_json(str(getattr(response, "content", "")))  # 从回复里容错解析 JSON
    if isinstance(data, dict):       # 解析出对象
        src = _clean_token(str(data.get("source", "")))
        dst = _clean_token(str(data.get("destination", "")))
        if src and dst:
            return src, dst, False   # 模型抽取默认不强制目录语义
    return None


def run_file_agent(                  # 执行一个“移动文件”步骤，返回结构化结果
    model: BaseChatModel,
    workspace: Workspace,
    step: dict[str, Any],
    *,
    allow_overwrite: bool = False,   # 是否允许覆盖已存在的目标（默认否）
) -> dict[str, Any]:
    """执行一个“移动文件”步骤，返回结构化结果。"""
    instruction = str(step.get("instruction", ""))  # 该步骤指令
    clause = _focus_clause(instruction)              # 聚焦到含移动关键词的子句

    # 源/目标优先用确定性正则抽取；正则失败才退回模型抽取
    extracted = _regex_extract(clause) or _regex_extract(instruction)  # 先聚焦子句、再退回整条
    extracted_by = "regex"           # 记录来源：正则
    if extracted is None:            # 正则都失败 → 让模型兜底抽取
        extracted = _llm_extract(model, clause)
        extracted_by = "llm"
    if extracted is None:            # 仍失败 → 返回“识别失败”
        return {
            "ok": False,
            "result": "未能从指令中识别出“源文件”和“目标位置”。",
            "error": "extract_failed",
            "artifact": "",
            "extracted_by": "none",
        }

    source, destination, dest_is_dir = extracted  # 拆出源/目标/目录标志
    if dest_is_dir and not destination.endswith(("/", "\\")):  # 是目录则补分隔符，让沙箱按“移动进目录”处理
        destination = destination + "/"

    # 真正的移动动作永远由 Python(shutil) 完成，绝不经过模型
    move_result = workspace.move_file(source, destination, allow_overwrite=allow_overwrite)
    if move_result.get("ok"):        # 移动成功
        return {
            "ok": True,
            "result": move_result.get("message", "移动完成"),
            "artifact": move_result.get("to", ""),
            "detail": move_result,
            "extracted_by": extracted_by,  # regex=正则抽取；llm=模型抽取（仅源/目标字符串）
        }
    return {                         # 移动失败（源不存在/越界/目标已存在等）
        "ok": False,
        "result": move_result.get("error", "移动失败"),
        "error": move_result.get("error", "move_failed"),
        "artifact": "",
        "detail": move_result,
        "extracted_by": extracted_by,
    }
