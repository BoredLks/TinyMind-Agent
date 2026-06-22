"""MiniMind Plan-and-Execute 多智能体 —— 命令行入口。

把仓库里本地部署的 MiniMind 当作 LLM，用 LangGraph 实现“先规划、再执行、可重试”的
多智能体，完成三类任务：移动文件 / 编写简单 Python 代码 / 编写小故事。

示例：
    python run_agent.py "写一个快速排序的Python函数"
    python run_agent.py "把 inbox/note.txt 移动到 archive 目录"
    python run_agent.py "写一个关于小猫的睡前故事"
    python run_agent.py "写一个冒泡排序函数，然后写一个关于程序员的小故事，最后把 inbox/data.csv 移动到 archive"

更多用法见 AGENT_README.md。
"""

from minimind_agent.cli import main  # 真正的 CLI 逻辑都在 minimind_agent/cli.py 里，这里只是薄入口

if __name__ == "__main__":           # 仅当作为脚本直接运行（python run_agent.py ...）时执行
    raise SystemExit(main())         # 用 main() 的返回值（0 成功 / 1 缺任务）作为进程退出码
