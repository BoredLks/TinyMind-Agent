"""把各节点装配成可运行的 Plan-and-Execute StateGraph。

    START → planner → {file_agent | code_agent | story_agent} ↺ → replan → final → END

外层只有“规划 + 执行循环 + 复核重试”，三个专家以独立节点出现，执行循环每轮处理一个待办步骤，
直到没有待办步骤时进入 replan 复核（失败则重试），最后由 final 汇总。
"""

from __future__ import annotations  # 启用延迟注解求值

from langgraph.graph import END, START, StateGraph  # 终点/起点占位 + 状态图类

from minimind_agent.graph.nodes import (  # 导入各节点与路由函数
    code_agent_node,
    file_agent_node,
    final_node,
    planner_node,
    replan_node,
    route_after_replan,
    route_after_step,
    story_agent_node,
)
from minimind_agent.graph.state import PlanExecuteState  # 图状态类型


def build_workflow():                # 构建并编译整张图
    """构建并编译 Plan-and-Execute 图。"""
    graph = StateGraph(PlanExecuteState)  # 以 PlanExecuteState 为状态类型创建状态图

    graph.add_node("planner", planner_node)        # 规划节点
    graph.add_node("file_agent", file_agent_node)  # 文件专家节点
    graph.add_node("code_agent", code_agent_node)  # 代码专家节点
    graph.add_node("story_agent", story_agent_node)  # 故事专家节点
    graph.add_node("replan", replan_node)          # 复核/重试节点
    graph.add_node("final", final_node)            # 汇总节点

    graph.add_edge(START, "planner")               # 起点 → 规划

    # 规划后 / 每个专家执行后，都用同一个路由决定“下一个待办步骤去哪”，没有待办则去 replan
    step_routes = {                  # 路由函数返回值 → 目标节点 的映射
        "file_agent": "file_agent",
        "code_agent": "code_agent",
        "story_agent": "story_agent",
        "replan": "replan",
    }
    graph.add_conditional_edges("planner", route_after_step, step_routes)  # 规划后的条件跳转
    for node in ("file_agent", "code_agent", "story_agent"):  # 每个专家执行后也用同一路由（形成执行循环）
        graph.add_conditional_edges(node, route_after_step, step_routes)

    # 复核后：有重试出的待办就回去执行，否则收尾
    graph.add_conditional_edges(
        "replan",
        route_after_replan,
        {
            "file_agent": "file_agent",
            "code_agent": "code_agent",
            "story_agent": "story_agent",
            "final": "final",
        },
    )
    graph.add_edge("final", END)     # 汇总 → 结束

    return graph.compile()           # 编译成可运行的图
