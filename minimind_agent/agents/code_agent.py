"""code 专家：编写简单 Python 代码。

这是最契合 MiniMind 强项的任务（其 SFT 语料里就有“用 Python 写斐波那契函数”一类）。
流程：LLM 生成 → 抽取代码块 → 落盘到 workspace/code/ → ``py_compile`` 语法校验
（默认 **只校验不执行**，更安全；如显式开启 allow_exec 才会在子进程里运行）。
"""

from __future__ import annotations  # 启用延迟注解求值

import os                            # 拆分路径、判断文件是否存在
import re                            # 正则：抽取代码块、提取函数/类名
from typing import Any               # 类型标注

from langchain_core.language_models.chat_models import BaseChatModel  # 模型类型
from langchain_core.messages import HumanMessage, SystemMessage       # 构造对话消息

from minimind_agent.prompts import CODE_SYSTEM, CODE_USER  # 写代码的提示词
from minimind_agent.workspace import Workspace             # 沙箱（落盘 + 语法校验）


def _extract_code(text: str) -> str:  # 从模型回复里抽取 Python 代码
    """从模型回复里抽取 Python 代码：优先 ``` 代码块，其次看是否整体就是代码。"""
    match = re.search(r"```(?:python|py)?\s*\n?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)  # 匹配 ```python...```
    if match:
        return match.group(1).strip("\n")  # 取代码块内容
    # 没有围栏：若包含明显的代码结构，就把整段当作代码
    if re.search(r"(^|\n)\s*(def |class |import |from |print\(|for |while |if )", text):
        return text.strip()
    return ""                        # 看不出是代码 → 返回空（后续走兜底）


def _filename_from_code(code: str) -> str:  # 根据代码里的函数/类名给文件命名
    """根据第一个函数/类名给文件命名，否则用 solution.py。"""
    match = re.search(r"\n\s*def\s+([A-Za-z_]\w*)", "\n" + code)  # 找第一个 def 名
    if match:
        return f"{match.group(1)}.py"
    match = re.search(r"\n\s*class\s+([A-Za-z_]\w*)", "\n" + code)  # 否则找第一个 class 名
    if match:
        return f"{match.group(1).lower()}.py"
    return "solution.py"             # 都没有就用默认名


def _unique_rel_path(workspace: Workspace, rel: str) -> str:  # 目标已存在则自动加序号，避免覆盖
    """若目标已存在，自动加序号，避免覆盖上一步的产物。"""
    if not (workspace.root / rel).exists():  # 不存在直接用
        return rel
    head, tail = os.path.split(rel)  # 拆成目录 + 文件名
    stem, ext = os.path.splitext(tail)  # 文件名拆成主名 + 扩展名
    index = 2                        # 从 _2 开始试
    while True:
        candidate = f"{head}/{stem}_{index}{ext}" if head else f"{stem}_{index}{ext}"  # 拼候选名
        if not (workspace.root / candidate).exists():  # 找到不冲突的名字
            return candidate
        index += 1


def _fallback_code(instruction: str) -> str:  # 模型没给可用代码时的兜底占位脚本
    """模型没给出可用代码时的兜底：写一个可编译的占位脚本，保留需求说明。"""
    safe = instruction.replace("\n", " ").strip()[:120]  # 取指令前 120 字写进注释
    return (
        f"# 需求: {safe}\n"
        "# 说明: 模型本次未生成可用代码，这是一个可运行的占位脚本。\n\n"
        "def main():\n"
        '    print("TODO: 实现上述需求")\n\n\n'
        'if __name__ == "__main__":\n'
        "    main()\n"
    )


def run_code_agent(                  # 执行一个“写 Python 代码”步骤，返回结构化结果
    model: BaseChatModel,
    workspace: Workspace,
    step: dict[str, Any],
    *,
    allow_exec: bool = False,        # 是否真正执行生成的代码（默认仅语法校验）
    exec_timeout: int = 10,          # 允许执行时的超时
) -> dict[str, Any]:
    """执行一个“写 Python 代码”步骤，返回结构化结果。"""
    instruction = str(step.get("instruction", ""))  # 该步骤指令
    try:
        response = model.invoke(     # 让模型按需求写代码
            [
                SystemMessage(content=CODE_SYSTEM),
                HumanMessage(content=CODE_USER.format(instruction=instruction)),
            ]
        )
        raw = str(getattr(response, "content", ""))  # 取模型回复文本
    except Exception as exc:  # 模型调用失败也要给出可用产物
        raw = ""
        model_error = f"{type(exc).__name__}: {exc}"  # 记录错误，写进兜底说明
    else:
        model_error = ""

    code = _extract_code(raw)        # 从回复里抽取代码
    used_fallback = False            # 是否用了占位兜底
    if not code.strip():             # 没抽到可用代码 → 写占位脚本
        code = _fallback_code(instruction)
        used_fallback = True

    rel = _unique_rel_path(workspace, f"code/{_filename_from_code(code)}")  # 计算落盘相对路径（不覆盖）
    write_result = workspace.write_text(rel, code)  # 写文件
    if not write_result.get("ok"):   # 写失败（极少见，如越界）
        return {"ok": False, "result": write_result.get("error", "写文件失败"), "artifact": "", "error": "write_failed"}

    artifact = write_result.get("path", rel)  # 产物路径
    check = workspace.syntax_check(rel)       # 语法校验（不执行）
    notes = []                       # 结果说明片段
    if used_fallback:                # 兜底时注明（这正是“来源=兜底”的依据）
        notes.append("模型未给出可用代码，已写入占位脚本" + (f"（{model_error}）" if model_error else ""))
    notes.append("语法校验通过" if check.get("ok") else f"语法校验失败：{check.get('error', '')}")  # 校验结论

    run_info: dict[str, Any] | None = None  # 执行信息（默认不执行）
    if allow_exec and check.get("ok"):  # 显式允许且语法通过 → 才在子进程里执行
        run_info = workspace.run_python(rel, timeout=exec_timeout)
        notes.append("执行成功" if run_info.get("ok") else f"执行失败：{run_info.get('stderr') or run_info.get('error', '')}")

    # 只要文件成功写出，就视为该步“完成”（语法/执行情况写进结果说明）
    return {
        "ok": True,
        "result": f"已生成 {artifact}；" + "；".join(notes),
        "artifact": artifact,
        "syntax_ok": bool(check.get("ok")),
        "executed": run_info,
        "used_fallback": used_fallback,  # True=代码是Python占位兜底；False=代码来自模型
    }
