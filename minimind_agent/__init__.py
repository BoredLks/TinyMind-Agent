"""minimind_agent —— 以本地 MiniMind 大模型为大脑的 Plan-and-Execute 多智能体。

本包参考 MokioAgent 的实现思路（LangGraph 编排 + 多专家 Agent + OpenAI 兼容 LLM 接口），
把仓库里本地部署的 MiniMind 模型当作整个 Agent 的 LLM，用 LangGraph 实现一个
“先规划、再执行、可重规划”的多智能体系统。三个专家 Agent 分别负责：

- file_agent ：移动文件（确定性 shutil 实现，限制在 workspace 沙箱内）
- code_agent ：编写简单 Python 代码（LLM 生成 → 落盘 → py_compile 语法校验）
- story_agent：编写小故事（LLM 生成 → 落盘）

对外主要入口见 ``minimind_agent.graph.workflow.build_workflow`` 与 ``minimind_agent.cli``。
"""

__all__ = ["__version__"]  # 本包对外导出的公共名字

__version__ = "0.1.0"  # 包版本号
