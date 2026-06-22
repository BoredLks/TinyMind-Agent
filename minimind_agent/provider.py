"""LLM 后端工厂：根据配置返回一个 LangChain ChatModel。

对应 MokioAgent 的 ``providers/openai_provider.py``，但这里支持三种后端：

- ``local`` ：进程内 MiniMind（``MiniMindChatModel``）。
- ``server``：``ChatOpenAI`` 连接 MiniMind 的 OpenAI 兼容服务（serve_openai_api.py）。
- ``mock``  ：确定性假模型，便于在无 torch / 无服务时跑通整张图与单测。

所有节点都通过 ``create_model(cfg)`` 取模型，并按后端缓存，避免 local 模式反复加载权重。
"""

from __future__ import annotations  # 启用延迟注解求值

import json                          # mock 模型把计划/抽取结果序列化成 JSON 字符串
import re                            # 预留（当前文件未直接使用，保持与原导入一致）
from typing import Any, Optional     # 类型标注

from langchain_core.language_models.chat_models import BaseChatModel  # ChatModel 基类（三种后端都实现它）
from langchain_core.messages import AIMessage, BaseMessage            # AI 消息 / 消息基类
from langchain_core.outputs import ChatGeneration, ChatResult         # ChatModel._generate 的标准返回容器

from minimind_agent.config import AgentConfig  # 运行配置


# 按后端缓存模型实例（local 权重只加载一次）
_MODEL_CACHE: dict[tuple, BaseChatModel] = {}  # 键=后端及其关键参数，值=已创建的模型实例


def create_model(cfg: AgentConfig) -> BaseChatModel:  # 工厂函数：按配置创建（或复用）一个 ChatModel
    """按配置创建（或复用）一个 ChatModel。"""
    if cfg.llm_mode == "mock":       # mock 后端：缓存键只需一个标记
        key: tuple = ("mock",)
    elif cfg.llm_mode == "server":   # server 后端：地址/模型名/温度不同则算不同实例
        key = ("server", cfg.base_url, cfg.model_name, cfg.temperature)
    else:  # local                   # local 后端：权重/结构/设备/温度不同则算不同实例
        key = ("local", cfg.weight, cfg.hidden_size, cfg.num_hidden_layers, cfg.use_moe, cfg.device, cfg.temperature)

    if key in _MODEL_CACHE:          # 命中缓存直接复用（避免重复加载权重/重建客户端）
        return _MODEL_CACHE[key]

    if cfg.llm_mode == "mock":       # —— 创建 mock 假模型 ——
        model: BaseChatModel = MockChatModel()
    elif cfg.llm_mode == "server":   # —— 创建 server 模式：ChatOpenAI 连本地 OpenAI 兼容服务 ——
        from langchain_openai import ChatOpenAI  # 惰性导入

        model = ChatOpenAI(
            model=cfg.model_name,            # 请求里的 model 字段
            api_key=cfg.api_key,             # 本地服务一般不校验，占位即可
            base_url=cfg.base_url,           # 服务地址（默认 serve_openai_api.py 的 :8998/v1）
            temperature=cfg.temperature,     # 采样温度
            max_tokens=cfg.max_new_tokens,   # 最大生成长度
        )
    else:                            # —— 创建 local 模式：进程内 MiniMind ——
        from minimind_agent.minimind_chat import MiniMindChatModel  # 惰性导入（避免无谓引入 torch）

        model = MiniMindChatModel(
            weight=cfg.weight,                       # 权重前缀（full_sft 等）
            hidden_size=cfg.hidden_size,             # 隐藏维度
            num_hidden_layers=cfg.num_hidden_layers,  # 层数
            use_moe=cfg.use_moe,                     # 是否 MoE
            device=cfg.device,                       # 设备（None=自动）
            temperature=cfg.temperature,             # 温度
            top_p=cfg.top_p,                         # 核采样阈值
            max_new_tokens=cfg.max_new_tokens,       # 最大生成长度
        )

    _MODEL_CACHE[key] = model        # 存入缓存
    return model                     # 返回模型实例


def reset_model_cache() -> None:     # 清空缓存（主要给测试用，避免不同用例互相影响）
    """清空模型缓存（主要给测试用）。"""
    _MODEL_CACHE.clear()


# ---------------------------------------------------------------------------
# MockChatModel：确定性假模型
# ---------------------------------------------------------------------------
def _mock_plan(task: str) -> list[dict[str, str]]:  # 用关键词把任务粗分成若干步骤（mock 模拟“LLM 规划”）
    """用关键词把任务粗分成若干步骤（仅供 mock 模拟 LLM 规划输出）。"""
    text = task.lower()              # 转小写便于匹配英文关键词
    steps: list[tuple[int, dict[str, str]]] = []  # (排序位置, 步骤) 列表
    file_kw = ("移动", "移到", "move", "mv ", "归档", "重命名")  # 文件类关键词
    code_kw = ("python", "代码", "函数", "脚本", "程序", "code", "算法")  # 代码类关键词
    story_kw = ("故事", "story", "童话", "小说", "作文")  # 故事类关键词
    if any(k in task or k in text for k in file_kw):  # 含文件关键词 → 加一个 file 步骤
        steps.append((task.find("移动") if "移动" in task else 0, {"agent": "file", "instruction": task}))
    if any(k in task or k in text for k in code_kw):  # 含代码关键词 → 加一个 code 步骤
        steps.append((1, {"agent": "code", "instruction": task}))
    if any(k in task or k in text for k in story_kw):  # 含故事关键词 → 加一个 story 步骤
        steps.append((2, {"agent": "story", "instruction": task}))
    if not steps:                    # 一个都没匹配 → 兜底给一个 story 步骤
        steps.append((0, {"agent": "story", "instruction": task}))
    return [s for _, s in sorted(steps, key=lambda x: x[0])]  # 按位置排序后返回纯步骤列表


def _mock_response(messages: list[BaseMessage]) -> str:  # 根据 system 提示判断在问什么，返回合理假回复
    """根据 system 提示词判断当前在问什么，返回合理的假回复。"""
    system_text = ""                 # 收集 system 提示文本
    user_text = ""                   # 收集 user 输入文本
    for message in messages:
        if message.type == "system":
            system_text = str(message.content)
        elif message.type == "human":
            user_text = str(message.content)

    if "任务规划助手" in system_text:  # 在做“规划” → 返回 JSON 计划
        task = user_text.split("用户任务：", 1)[-1].split("\n", 1)[0].strip()  # 从 user 输入里抠出任务文本
        return json.dumps(_mock_plan(task), ensure_ascii=False)

    if "Python 程序员" in system_text:  # 在做“写代码” → 返回一段一定能编译的占位代码
        instruction = user_text.replace("\n", " ")[:80]  # 取指令前 80 字写进注释
        code = (
            f"# 需求: {instruction}\n\n"
            "def main():\n"
            '    print("hello from minimind code agent")\n\n\n'
            'if __name__ == "__main__":\n'
            "    main()\n"
        )
        return f"```python\n{code}```"  # 包成 ```python 代码块（与真实模型输出格式一致）

    if "会写故事的作家" in system_text:  # 在做“写故事” → 返回一段固定小故事
        return (
            "从前有一个叫小敏的程序员，他训练了一个很小很小的语言模型。"
            "模型虽然只有几千万参数，却努力地学习写代码、讲故事。"
            "有一天，它第一次独立完成了任务，小敏开心地笑了。"
        )

    if "抽取" in system_text:        # 在做“源/目标抽取” → 返回空 JSON（实际靠正则，故 mock 不需要真抽）
        return json.dumps({"source": "", "destination": ""}, ensure_ascii=False)

    return user_text[:200]           # 其它情况：原样回显用户输入前 200 字


class MockChatModel(BaseChatModel):  # 确定性假模型：不依赖外部服务/torch，用于测试整张图
    """不依赖任何外部服务 / torch 的确定性 ChatModel，用于测试整张图。"""

    @property
    def _llm_type(self) -> str:      # ChatModel 要求实现：返回模型类型标识
        return "minimind-mock"

    def _generate(                   # ChatModel 的核心方法：给定消息返回一次生成结果
        self,
        messages: list[BaseMessage],  # 输入消息列表
        stop: Optional[list[str]] = None,  # 停止词（mock 不使用）
        run_manager: Any = None,     # 回调管理器（mock 不使用）
        **kwargs: Any,
    ) -> ChatResult:
        text = _mock_response(messages)  # 计算假回复文本
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])  # 包成标准返回
