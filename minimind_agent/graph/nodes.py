"""Plan-and-Execute 的各个节点与路由函数。

整体流程：

    planner  →  (file/code/story)_agent  ↺  →  replan  →  final

- planner ：让 MiniMind 产出 JSON 计划；解析失败则用确定性关键词分类器兜底，保证一定有可执行计划。
- 三个 *_agent 节点：每次处理“第一个待办步骤”，把它交给对应专家执行，并更新计划状态。
- replan  ：若有步骤失败且未超过重试上限，则把失败步骤重置为待办再跑一轮。
- final   ：汇总计划、各步结果与 workspace 产物（不调用模型）。
"""

from __future__ import annotations  # 启用延迟注解求值

import re                            # 正则：把任务切成子句
from typing import Any, Optional     # 类型标注

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # 三种消息类型
from langgraph.config import get_stream_writer  # 取“自定义事件写出器”（流式展示）

from minimind_agent.config import AgentConfig  # 运行配置
from minimind_agent.agents import run_code_agent, run_file_agent, run_story_agent  # 三个专家
from minimind_agent.graph.state import PlanExecuteState, PlanStep  # 图状态与步骤类型
from minimind_agent.parsing import extract_json  # 容错解析模型输出的 JSON
from minimind_agent.prompts import PLANNER_SYSTEM, PLANNER_USER  # 规划提示词
from minimind_agent.provider import create_model  # 取（缓存的）模型
from minimind_agent.workspace import Workspace  # 沙箱


VALID_AGENTS = ("file", "code", "story")  # 合法的专家名

# 把任务切成子句的分隔符（顺序连接词 + 标点）
_CLAUSE_SPLIT = re.compile(r"然后|接着|之后|再|并且|并|最后|最终|[，,。；;、\n]")

# 各专家的关键词（注意检测顺序：file → story → code，避免“程序员的故事”被误判为 code）
_FILE_KW = ("移动", "移到", "移至", "挪", "归档", "重命名", "放到", "放进", "move ", "mv ")  # 文件类
_STORY_KW = ("故事", "小说", "童话", "作文", "睡前", "story")  # 故事类
_CODE_KW = ("python", "代码", "函数", "脚本", "算法", "程序", "计算", "排序", "爬虫", "code", "def ")  # 代码类


def _get_writer():                   # 取 LangGraph 事件写出器；不在图里运行时返回空操作
    """取 LangGraph 自定义事件写出器；不在图运行环境时返回空操作。"""
    try:
        return get_stream_writer()   # 在 .stream() 上下文里可用
    except RuntimeError:             # 直接 .invoke() 等场景下没有 writer
        return lambda _event: None   # 返回一个吃掉事件的空函数


def _cfg(state: PlanExecuteState) -> AgentConfig:  # 从状态里取配置；缺失则用环境默认
    cfg = state.get("config")
    return cfg if isinstance(cfg, AgentConfig) else AgentConfig.from_env()


# --------------------------------------------------------------------------- #
# 计划相关：检测专家、确定性兜底分类、归一化
# --------------------------------------------------------------------------- #
def _detect_agent(text: str) -> Optional[str]:  # 按关键词判断一段文字该归哪个专家
    """根据关键词判断一段文字该交给哪个专家；都不匹配返回 None。"""
    lowered = text.lower()           # 转小写匹配英文关键词
    if any(k in text or k in lowered for k in _FILE_KW):  # 先判 file（“移动”等最明确）
        return "file"
    if any(k in text or k in lowered for k in _STORY_KW):  # 再判 story（“故事”，避免被 code 抢走）
        return "story"
    if any(k in text or k in lowered for k in _CODE_KW):  # 最后判 code
        return "code"
    return None                      # 都不匹配


def _classify_fallback(task: str) -> list[PlanStep]:  # 确定性兜底规划：按子句切分，逐句归专家
    """确定性兜底规划：按子句切分任务，逐句归到对应专家。"""
    pieces = [p.strip() for p in _CLAUSE_SPLIT.split(task) if p.strip()]  # 切子句并去空
    raw: list[dict[str, str]] = []   # 收集 {agent, instruction}
    buffer: list[str] = []           # 暂存“暂无关键词”的子句，等下一个有关键词的子句来吸收
    for piece in pieces:
        agent = _detect_agent(piece)  # 这段属于哪个专家
        if agent is None:            # 没关键词 → 先缓冲
            buffer.append(piece)
            continue
        instruction = "，".join(buffer + [piece]) if buffer else piece  # 把缓冲的前置子句并进来
        buffer = []                  # 清空缓冲
        raw.append({"agent": agent, "instruction": instruction})
    if buffer:  # 处理结尾没有关键词的残句
        if raw:                      # 有步骤就并到最后一步
            raw[-1]["instruction"] += "，" + "，".join(buffer)
        else:                        # 整条都没关键词 → 兜底成一步（默认 story）
            raw.append({"agent": _detect_agent(task) or "story", "instruction": task})
    if not raw:                      # 兜底：保证至少一步
        raw.append({"agent": _detect_agent(task) or "story", "instruction": task})
    return _normalize_plan(raw)      # 统一归一化成 PlanStep


def _normalize_plan(raw_steps: Any) -> list[PlanStep]:  # 把任意步骤列表归一化成合法 PlanStep 列表
    """把任意（模型/兜底产生的）步骤列表归一化成合法的 PlanStep 列表。"""
    steps: list[PlanStep] = []
    if not isinstance(raw_steps, list):  # 不是列表（模型乱答）→ 空计划
        return steps
    for index, item in enumerate(raw_steps, start=1):  # 逐项处理，编号从 1 起
        if not isinstance(item, dict):  # 跳过非字典项
            continue
        instruction = str(item.get("instruction", "")).strip()  # 取指令
        if not instruction:          # 指令为空则跳过
            continue
        agent = str(item.get("agent", "")).strip().lower()  # 取专家名
        if agent not in VALID_AGENTS:  # 非法专家名 → 用关键词重新推断（默认 story）
            agent = _detect_agent(instruction) or "story"
        steps.append(                # 组装一个标准步骤（初始 status=pending）
            {
                "id": f"step-{index}",
                "agent": agent,
                "instruction": instruction,
                "status": "pending",
                "result": "",
                "artifact": "",
                "error": "",
            }
        )
    return steps


# --------------------------------------------------------------------------- #
# 节点：planner
# --------------------------------------------------------------------------- #
def planner_node(state: PlanExecuteState) -> dict[str, Any]:  # 规划节点：模型出计划，失败则兜底
    """规划节点：先让 MiniMind 产出计划，解析失败再用确定性分类兜底。"""
    cfg = _cfg(state)                # 取配置
    task = str(state.get("task", "")).strip()  # 用户任务
    writer = _get_writer()           # 事件写出器

    # 准备 workspace（并按需生成演示样例，让“移动文件”有东西可移）
    workspace = Workspace(cfg.workspace)
    if cfg.seed_samples:
        workspace.seed_samples()

    plan: list[PlanStep] = []        # 计划
    plan_source = "fallback"         # 计划来源（默认兜底）
    try:
        model = create_model(cfg)    # 取模型
        response = model.invoke(     # 让模型产出 JSON 计划
            [
                SystemMessage(content=PLANNER_SYSTEM),
                HumanMessage(content=PLANNER_USER.format(task=task)),
            ]
        )
        parsed = extract_json(str(getattr(response, "content", "")))  # 容错解析
        plan = _normalize_plan(parsed)  # 归一化
        if plan:                     # 解析出可用计划
            plan_source = "llm"
    except Exception as exc:  # 模型不可用也不能让规划崩掉
        writer({"type": "planner_warning", "message": f"LLM 规划失败，改用兜底分类：{type(exc).__name__}: {exc}"})

    if not plan:  # 模型没给出可用计划 → 确定性兜底
        plan = _classify_fallback(task)
        plan_source = "fallback"

    writer({"type": "plan", "source": plan_source, "backend": cfg.backend_label(), "plan": plan})  # 发出“计划”事件
    return {                         # 写回状态：计划 + 来源 + 初始化执行相关字段
        "plan": plan,
        "plan_source": plan_source,
        "past_steps": [],
        "replans": 0,
        "max_replans": cfg.max_replans,
        "messages": [AIMessage(content=f"已生成计划（来源：{plan_source}），共 {len(plan)} 步。")],
    }


# --------------------------------------------------------------------------- #
# 节点：三个专家（共用 _run_step；以独立节点呈现，便于在图里看清“三个 Agent”）
# --------------------------------------------------------------------------- #
def _next_pending_index(plan: list[PlanStep]) -> Optional[int]:  # 找第一个 pending 步骤的下标
    for index, step in enumerate(plan):
        if step.get("status") == "pending":
            return index
    return None                      # 没有待办


def _run_step(state: PlanExecuteState) -> dict[str, Any]:  # 执行第一个待办步骤（按 agent 分派）
    """执行“第一个待办步骤”：按 step.agent 分派给对应专家。"""
    cfg = _cfg(state)                # 配置
    workspace = Workspace(cfg.workspace)  # 沙箱
    model = create_model(cfg)        # 模型
    writer = _get_writer()           # 事件写出器

    plan = [dict(step) for step in state.get("plan", [])]  # 复制一份计划再改
    index = _next_pending_index(plan)  # 找待办
    if index is None:                # 没有待办则空操作
        return {}
    step = plan[index]               # 当前要执行的步骤
    agent = step.get("agent", "story")  # 它属于哪个专家
    writer({"type": "step_start", "id": step.get("id"), "agent": agent, "instruction": step.get("instruction", "")})  # 发“开始”事件

    if agent == "file":              # 分派给文件专家
        result = run_file_agent(model, workspace, step)
    elif agent == "code":            # 分派给代码专家（透传执行开关）
        result = run_code_agent(model, workspace, step, allow_exec=cfg.allow_exec, exec_timeout=cfg.exec_timeout)
    else:                            # 其余给故事专家
        result = run_story_agent(model, workspace, step)

    source = _result_source(cfg, agent, result)  # 计算“产物来源”标签（模型/兜底/正则）
    step["status"] = "done" if result.get("ok") else "failed"  # 更新状态
    step["result"] = str(result.get("result", ""))  # 结果摘要
    step["artifact"] = str(result.get("artifact", ""))  # 产物路径
    step["error"] = str(result.get("error", ""))  # 失败原因
    step["source"] = source          # 来源
    plan[index] = step               # 写回该步骤

    past = list(state.get("past_steps", []))  # 累积“已执行步骤”记录
    past.append(
        {
            "id": step.get("id"),
            "agent": agent,
            "ok": bool(result.get("ok")),
            "result": step["result"],
            "artifact": step["artifact"],
            "source": source,
        }
    )
    writer(                          # 发“步骤结果”事件（网页/CLI 据此渲染来源徽章）
        {
            "type": "step_result",
            "id": step.get("id"),
            "agent": agent,
            "ok": bool(result.get("ok")),
            "result": step["result"],
            "artifact": step["artifact"],
            "source": source,
        }
    )
    return {"plan": plan, "past_steps": past}  # 写回状态（plan 全量替换；顺序执行无并发问题）


def _result_source(cfg: AgentConfig, agent: str, result: dict[str, Any]) -> str:  # 计算产物的真实来源标签
    """根据专家回报，标注该步产物的真实来源（模型 / 兜底 / 正则）。"""
    backend = cfg.backend_label()    # 后端名（MiniMind / 假模型 等）
    if agent in ("code", "story"):   # 代码/故事：看是否兜底
        return "兜底(Python占位)" if result.get("used_fallback") else f"内容由 {backend} 生成"
    if agent == "file":              # 文件：移动永远是 Python；区分源/目标是否由模型抽取
        if result.get("extracted_by") == "llm":
            return f"移动=Python(shutil)；源/目标由 {backend} 抽取"
        return "全程 Python(正则抽取源/目标 + shutil 移动)"
    return backend                   # 兜底


def file_agent_node(state: PlanExecuteState) -> dict[str, Any]:  # 文件专家节点（共用 _run_step）
    return _run_step(state)


def code_agent_node(state: PlanExecuteState) -> dict[str, Any]:  # 代码专家节点
    return _run_step(state)


def story_agent_node(state: PlanExecuteState) -> dict[str, Any]:  # 故事专家节点
    return _run_step(state)


# --------------------------------------------------------------------------- #
# 路由：决定下一步去哪个节点
# --------------------------------------------------------------------------- #
def route_after_step(state: PlanExecuteState) -> str:  # 规划/执行后：有待办去对应专家，否则去 replan
    """执行/规划之后：还有待办步骤就去对应专家，否则进入 replan 复核。"""
    plan = state.get("plan", [])
    index = _next_pending_index(plan)
    if index is None:                # 无待办 → 复核
        return "replan"
    return f"{plan[index].get('agent', 'story')}_agent"  # 否则去“<agent>_agent”节点


def route_after_replan(state: PlanExecuteState) -> str:  # 复核后：有重试出的待办回去执行，否则收尾
    """复核之后：若 replan 重置出新的待办步骤就回去执行，否则进入 final 收尾。"""
    plan = state.get("plan", [])
    index = _next_pending_index(plan)
    if index is None:                # 无待办 → 收尾
        return "final"
    return f"{plan[index].get('agent', 'story')}_agent"  # 否则回去执行


# --------------------------------------------------------------------------- #
# 节点：replan（失败重试）
# --------------------------------------------------------------------------- #
def replan_node(state: PlanExecuteState) -> dict[str, Any]:  # 复核：失败步骤重置为待办再试（受上限约束）
    """复核节点：把失败步骤重置为待办再试一轮（受 max_replans 限制）。"""
    writer = _get_writer()
    plan = [dict(step) for step in state.get("plan", [])]  # 复制计划
    failed = [step for step in plan if step.get("status") == "failed"]  # 失败步骤
    replans = int(state.get("replans", 0))      # 已重试轮数
    max_replans = int(state.get("max_replans", 1))  # 上限

    if failed and replans < max_replans:  # 有失败且未超限 → 重置失败步骤为待办
        for step in plan:
            if step.get("status") == "failed":
                step["status"] = "pending"
        replans += 1                 # 重试轮数 +1
        writer(                      # 发“重规划”事件
            {
                "type": "replan",
                "replans": replans,
                "retry": [step.get("id") for step in failed],
            }
        )
        return {"plan": plan, "replans": replans}  # 写回（路由会再去执行这些待办）

    # 无可重试项或已达上限：保持现状，交给 final
    return {"replans": replans}


# --------------------------------------------------------------------------- #
# 节点：final（汇总）
# --------------------------------------------------------------------------- #
def final_node(state: PlanExecuteState) -> dict[str, Any]:  # 收尾：拼人类可读总结（不调用模型）
    """收尾节点：把计划、各步结果与 workspace 产物拼成人类可读的总结。"""
    cfg = _cfg(state)
    workspace = Workspace(cfg.workspace)
    plan = state.get("plan", [])
    task = state.get("task", "")
    plan_source = state.get("plan_source", "fallback")
    backend = cfg.backend_label()
    plan_from = f"{backend} 规划" if plan_source == "llm" else "确定性关键词分类器(Python兜底)"  # 计划来源描述

    done = sum(1 for step in plan if step.get("status") == "done")    # 成功步数
    failed = sum(1 for step in plan if step.get("status") == "failed")  # 失败步数
    overall = "全部完成" if failed == 0 else f"完成 {done} 步，失败 {failed} 步"  # 总体结论

    icon = {"done": "[done]", "failed": "[fail]", "pending": "[todo]"}  # 状态 → ASCII 标记
    lines = [                        # 逐行拼总结
        "=== MiniMind Plan-and-Execute 多智能体 运行结果 ===",
        f"任务：{task}",
        f"LLM 后端：{backend}",
        f"计划来源：{plan_from}",
        f"总体：{overall}",
        "",
        "步骤明细（含每步产物来源）：",
    ]
    for step in plan:                # 每个步骤一段
        mark = icon.get(step.get("status", ""), "[?]")
        lines.append(f"  {mark} [{step.get('agent')}] {step.get('instruction')}")
        if step.get("result"):
            lines.append(f"      → {step.get('result')}")
        if step.get("source"):
            lines.append(f"      └ 来源: {step.get('source')}")

    files = workspace.list_files()   # 列出 workspace 产物
    lines.append("")
    lines.append(f"workspace（{workspace.root}）当前文件：")
    if files:
        lines.extend(f"  - {name}" for name in files)
    else:
        lines.append("  （空）")

    final_answer = "\n".join(lines)  # 合成最终文本
    _get_writer()({"type": "final", "answer": final_answer})  # 发“final”事件
    return {"final_answer": final_answer, "messages": [AIMessage(content=final_answer)]}  # 写回状态
