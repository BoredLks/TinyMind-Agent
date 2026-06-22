"""从模型输出里“尽力”抽取 JSON 的小工具。

小模型常把 JSON 混在解释文字 / ``` 代码块里，甚至不闭合。这里做容错抽取：
优先匹配 ``` 代码块，再退而求其次扫描第一个完整的 {...} 或 [...]。
"""

from __future__ import annotations  # 启用延迟注解求值（允许使用 X | None 等新式写法）

import json                          # 解析 JSON 字符串
from typing import Any               # 返回值类型不固定（dict / list / None），用 Any 标注


def _try_load(snippet: str) -> Any:  # 尝试把一段字符串解析为 JSON；失败返回 None（不抛异常）
    try:
        return json.loads(snippet)   # 解析成功则返回对应的 Python 对象（dict/list/...）
    except json.JSONDecodeError:     # 不是合法 JSON
        return None                  # 用 None 表示“这段解析不了”


def extract_json(text: str) -> Any:  # 对外主函数：从任意文本里抽出第一个可解析的 JSON 值
    """返回文本中第一个可解析的 JSON 值（dict 或 list），失败返回 None。"""
    if not text:                     # 空字符串/None 直接返回
        return None

    # 1) ```json ... ``` 或 ``` ... ``` 代码块
    fenced = _between(text, "```")   # 取第一对 ``` 之间的内容（模型常把 JSON 放代码块里）
    candidates = []                  # 候选片段列表，按优先级依次尝试
    if fenced is not None:           # 找到了代码块
        cleaned = fenced             # 代码块内容
        if cleaned.lower().startswith("json"):  # 形如 ```json 时，开头会多出 "json" 这个语言标记
            cleaned = cleaned[4:]    # 去掉这 4 个字符
        candidates.append(cleaned.strip())  # 去空白后作为首选候选

    # 2) 整段
    candidates.append(text)          # 退而求其次：把整段原文也作为候选

    for candidate in candidates:     # 逐个候选尝试
        value = _try_load(candidate)  # 先直接整段解析
        if value is not None:        # 成功就返回
            return value
        # 3) 扫描最外层的 [...] 或 {...}
        for open_ch, close_ch in (("[", "]"), ("{", "}")):  # 先试数组，再试对象
            start = candidate.find(open_ch)   # 第一个左括号位置
            end = candidate.rfind(close_ch)   # 最后一个右括号位置
            if start != -1 and end != -1 and end > start:  # 存在成对且顺序正确的括号
                value = _try_load(candidate[start : end + 1])  # 截取这段再解析
                if value is not None:
                    return value     # 成功就返回
    return None                      # 所有候选都失败


def _between(text: str, fence: str) -> str | None:  # 取第一对分隔符之间的内容（这里 fence 是 ```）
    """取第一对 ``` 之间的内容。"""
    first = text.find(fence)         # 第一个 ``` 的位置
    if first == -1:                  # 没有起始 ```
        return None
    second = text.find(fence, first + len(fence))  # 从起始之后再找第二个 ```
    if second == -1:                 # 没有闭合 ```
        return None
    return text[first + len(fence) : second]  # 返回两对 ``` 之间的纯内容
