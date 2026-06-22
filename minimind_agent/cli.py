"""命令行入口：解析参数 → 组装配置 → 运行 Plan-and-Execute 图 → 流式打印过程与结果。

用法示例：
    python run_agent.py "写一个冒泡排序的Python函数，然后写一个关于程序员的小故事"
    python run_agent.py --mode server "把 inbox/note.txt 移动到 archive 目录"
    python run_agent.py --mode mock   "..."   # 不需要 torch / 不需要服务，跑通流程

输出刻意使用纯 ASCII 结构标记 + 强制 UTF-8 stdout，避免在中文(GBK)控制台下崩溃。
"""

from __future__ import annotations  # 启用延迟注解求值

import argparse                      # 解析命令行参数
import sys                           # 访问 stdout/stderr、退出码
from typing import Any               # 类型标注


from minimind_agent.config import AgentConfig          # 运行配置
from minimind_agent.graph.workflow import build_workflow  # 编译好的图


def _ensure_utf8_stdout() -> None:   # 尽力把 stdout/stderr 切到 UTF-8，避免 GBK 控制台编码崩溃
    """尽力把 stdout/stderr 切到 UTF-8（errors=replace），避免 GBK 控制台下编码崩溃。"""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)          # 取流对象
        reconfigure = getattr(stream, "reconfigure", None)  # Python 3.7+ 才有 reconfigure
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")  # 切 UTF-8，坏字符用替换而非报错
            except Exception:  # pragma: no cover - 个别环境不支持
                pass


def _out(text: str = "") -> None:    # 安全打印：极端编码环境也不崩
    try:
        print(text)
    except Exception:  # pragma: no cover - 极端编码环境兜底
        print(text.encode("utf-8", "replace").decode("ascii", "replace"))  # 退化成 ASCII 安全输出


_AGENT_LABEL = {"file": "file_agent", "code": "code_agent", "story": "story_agent"}  # 专家名 → 显示标签


def _print_event(event: dict[str, Any]) -> None:  # 把图运行期事件渲染成一行行可读输出
    """把图运行期的自定义事件渲染成一行行可读输出（纯 ASCII 结构标记）。"""
    etype = event.get("type")        # 事件类型
    if etype == "plan":              # 计划事件：打印计划来源与各步
        plan_from = f"{event.get('backend')} 规划" if event.get("source") == "llm" else "确定性关键词分类器(Python兜底)"
        _out(f"\n[planner] 生成计划 (来源: {plan_from}):")
        for step in event.get("plan", []):
            _out(f"   - <{step.get('agent')}> {step.get('instruction')}")
    elif etype == "planner_warning":  # 规划告警
        _out(f"[planner] (warn) {event.get('message')}")
    elif etype == "step_start":      # 某步开始
        label = _AGENT_LABEL.get(event.get("agent", ""), event.get("agent", ""))
        _out(f"\n>> [{label}] 开始: {event.get('instruction')}")
    elif etype == "step_result":     # 某步结果（含来源标注）
        mark = "[OK]" if event.get("ok") else "[FAIL]"
        label = _AGENT_LABEL.get(event.get("agent", ""), "")
        _out(f"{mark} [{label}] -> {event.get('result')}")
        if event.get("source"):
            _out(f"       «来源: {event.get('source')}»")
    elif etype == "replan":          # 重试事件
        _out(f"\n[replan] 第 {event.get('replans')} 次重试, 重置步骤: {event.get('retry')}")
    elif etype == "final":
        pass  # 最终结果统一在最后打印


def run(task: str, **overrides: Any) -> str:  # 运行一次完整流程，返回最终总结文本
    """运行一次完整流程，返回最终总结文本。"""
    _ensure_utf8_stdout()            # 先把输出切 UTF-8
    cfg = AgentConfig.from_env(**overrides)  # 组装配置（CLI 参数覆盖环境默认）
    _out(
        f"启动 MiniMind Plan-and-Execute 多智能体 | LLM 后端: {cfg.backend_label()} | "
        f"workspace: {cfg.workspace}"
    )
    if cfg.llm_mode == "mock":       # mock 模式：醒目警告“产物是假数据”
        _out("!! 警告：当前是 mock 假模型模式，产物均为占位假数据，不是真实 MiniMind 输出。")
        _out("!! 要用真实模型请去掉 --mode mock（默认 local），或设置 MINIMIND_LLM_MODE=local。")
    workflow = build_workflow()      # 构建图
    inputs = {                       # 初始状态
        "task": task,
        "config": cfg,
        "messages": [],
        "max_replans": cfg.max_replans,
    }
    final_answer = ""                # 收集最终答案
    for mode, payload in workflow.stream(inputs, stream_mode=["updates", "custom"]):  # 双流：更新 + 自定义事件
        if mode == "custom":         # 自定义事件 → 实时打印
            _print_event(payload)
        elif isinstance(payload, dict):  # updates：捕获 final 节点产出的最终答案
            for update in payload.values():
                if isinstance(update, dict) and update.get("final_answer"):
                    final_answer = update["final_answer"]

    _out("\n" + "=" * 60)            # 分隔线
    _out(final_answer or "(无输出)")  # 打印最终总结
    _out("=" * 60)
    return final_answer


def build_arg_parser() -> argparse.ArgumentParser:  # 构建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="MiniMind Plan-and-Execute 多智能体（file / code / story 三专家）",
    )
    parser.add_argument("task", nargs="*", help="要完成的任务（可用引号包裹）")  # 位置参数：任务（可多段）
    parser.add_argument("--mode", dest="llm_mode", choices=["local", "server", "mock"], default=None,
                        help="LLM 后端：local=进程内权重，server=连接 serve_openai_api.py，mock=假模型")
    parser.add_argument("--workspace", default=None, help="工作区目录（所有文件操作的沙箱根）")
    parser.add_argument("--weight", default=None, help="local 模式权重前缀（默认 full_sft）")
    parser.add_argument("--hidden-size", dest="hidden_size", type=int, default=None, help="local 模式隐藏层维度")
    parser.add_argument("--device", default=None, help="local 模式设备（cuda/cpu，默认自动）")
    parser.add_argument("--base-url", dest="base_url", default=None, help="server 模式服务地址")
    parser.add_argument("--temperature", type=float, default=None, help="生成温度")
    parser.add_argument("--max-replans", dest="max_replans", type=int, default=None, help="失败后最多重试几轮")
    parser.add_argument("--allow-exec", dest="allow_exec", action="store_true", default=None,
                        help="允许 code_agent 真正执行生成的代码（默认仅语法校验）")
    parser.add_argument("--no-seed", dest="seed_samples", action="store_false", default=None,
                        help="不在 workspace 内生成演示样例文件")
    return parser


def main(argv: list[str] | None = None) -> int:  # 程序主入口
    _ensure_utf8_stdout()            # 切 UTF-8
    parser = build_arg_parser()      # 解析器
    args = parser.parse_args(argv)   # 解析参数
    task = " ".join(args.task).strip()  # 把多段任务拼成一句
    if not task:                     # 没给任务则打印帮助
        parser.print_help()
        _out('\n示例：python run_agent.py "写一个计算阶乘的Python函数"')
        return 1

    overrides = {                    # 把 CLI 参数收集成 overrides（None 的项稍后过滤掉）
        "llm_mode": args.llm_mode,
        "workspace": args.workspace,
        "weight": args.weight,
        "hidden_size": args.hidden_size,
        "device": args.device,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "max_replans": args.max_replans,
        "allow_exec": args.allow_exec,
        "seed_samples": args.seed_samples,
    }
    overrides = {key: value for key, value in overrides.items() if value is not None}  # 只保留显式指定的项
    run(task, **overrides)           # 运行
    return 0


if __name__ == "__main__":           # 作为脚本运行
    sys.exit(main())                 # 用返回值作为退出码
