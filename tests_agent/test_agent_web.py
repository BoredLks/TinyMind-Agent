"""agent_web.py 的无头测试：用 Streamlit 官方 AppTest 在模拟运行时里执行整个网页脚本，
确保（1）页面能正常加载、（2）mock 后端下点「运行」能跑通并产出文件，且全程无异常。

未安装 streamlit 时自动跳过（不影响其余 mock 测试）。从项目根运行：
    python -m pytest tests_agent/test_agent_web.py -q
"""

from __future__ import annotations  # 启用延迟注解求值

from pathlib import Path             # tmp_path 夹具

import pytest                        # 测试框架

pytest.importorskip("streamlit")  # 没装 streamlit 就跳过本文件
from streamlit.testing.v1 import AppTest  # noqa: E402  # Streamlit 官方无头测试工具

APP = str(Path(__file__).resolve().parents[1] / "agent_web.py")  # 网页脚本绝对路径


def test_app_loads_without_error():  # 用例 1：页面首次加载（未点运行）应无异常
    """页面首次加载（未点运行）应无异常。"""
    at = AppTest.from_file(APP, default_timeout=60).run()  # 在模拟运行时里执行脚本
    assert not at.exception, at.exception  # 不应有任何异常
    # Hero 头部（自定义 HTML）渲染出来了
    assert any("多智能体" in str(getattr(m, "value", "")) for m in at.markdown)  # markdown 里应含标题文字


def test_app_mock_run_produces_artifacts(tmp_path: Path):  # 用例 2：mock 模式点“运行”应产出文件且无异常
    """mock 后端下，设置任务并点击「运行」，应无异常并在 workspace 产出文件。"""
    at = AppTest.from_file(APP, default_timeout=180).run()  # 先跑一次让控件就绪
    assert not at.exception, at.exception

    at.radio(key="mode").set_value("mock")                  # 选 mock 后端
    at.text_input(key="workspace").set_value(str(tmp_path))  # 沙箱指到临时目录（不污染项目）
    at.checkbox(key="seed_samples").set_value(True)         # 生成演示样例（含 inbox/note.txt）
    at.text_area(key="task_input").set_value(               # 设置一个三类任务都覆盖的组合任务
        "写一个计算阶乘的Python函数，然后写一个关于小猫的故事，最后把 inbox/note.txt 移动到 archive 目录"
    )
    at.run()                         # 应用控件设置后重跑

    at.button(key="run_btn").click().run()  # 点击“运行智能体”并重跑（真正驱动整张图）
    assert not at.exception, at.exception   # 运行全程无异常

    # 三类产物都应落到 workspace 沙箱
    assert list((tmp_path / "code").glob("*.py")), "code_agent 应产出 .py"        # 代码产物
    assert list((tmp_path / "stories").glob("*.md")), "story_agent 应产出 .md"     # 故事产物
    assert (tmp_path / "archive" / "note.txt").exists(), "file_agent 应移动文件到 archive"  # 移动产物

    # session_state 里应记录了本次运行
    assert at.session_state["last_run"]["ok"] is True  # 运行成功标志
