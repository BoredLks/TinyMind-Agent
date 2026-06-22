"""三个专家 Agent：file（移动文件）、code（写 Python）、story（写小故事）。"""

from minimind_agent.agents.code_agent import run_code_agent  # 代码专家入口
from minimind_agent.agents.file_agent import run_file_agent  # 文件专家入口
from minimind_agent.agents.story_agent import run_story_agent  # 故事专家入口

__all__ = ["run_file_agent", "run_code_agent", "run_story_agent"]  # 对外导出三个专家函数
