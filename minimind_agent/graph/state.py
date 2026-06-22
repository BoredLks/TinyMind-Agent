"""Plan-and-Execute 图的共享状态。"""

from __future__ import annotations  # 启用延迟注解求值

from typing import Annotated, Any, TypedDict  # Annotated 给字段附加归约器；TypedDict 带类型的字典

from langchain_core.messages import BaseMessage  # 消息基类
from langgraph.graph import add_messages          # 消息合并归约器（节点返回新消息时智能合并）

from minimind_agent.config import AgentConfig     # 运行配置类型


class PlanStep(TypedDict, total=False):
    """计划中的单个步骤：交给某个专家执行一条指令。"""

    id: str            # 步骤编号，如 "step-1"
    agent: str         # 负责的专家：file / code / story
    instruction: str   # 交给专家的自然语言指令
    status: str        # pending / done / failed
    result: str        # 执行结果摘要
    artifact: str      # 产物（通常是生成/移动的文件路径）
    error: str         # 失败原因
    source: str        # 产物来源：真实模型后端名 / 兜底(Python占位) / 正则抽取 等


class PlanExecuteState(TypedDict, total=False):
    """贯穿整张图的状态：所有节点读写它。total=False 表示字段都可选。"""

    task: str                                       # 用户原始任务
    config: AgentConfig                             # 运行配置（含 workspace、LLM 后端等）
    messages: Annotated[list[BaseMessage], add_messages]  # 可选的消息记录（带智能合并）
    plan: list[PlanStep]                            # 计划（有序步骤）
    plan_source: str                                # 计划来源：llm / fallback
    past_steps: list[dict[str, Any]]                # 已执行步骤的结果记录
    replans: int                                    # 已重规划次数
    max_replans: int                                # 最大重规划次数
    final_answer: str                               # 最终人类可读总结
