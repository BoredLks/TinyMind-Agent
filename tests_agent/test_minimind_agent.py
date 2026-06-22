"""minimind_agent 的单元 + 端到端测试（全部使用 mock 后端，无需 torch / 无需服务）。

从项目根运行：  python -m pytest tests_agent -q
"""

from __future__ import annotations  # 启用延迟注解求值

from pathlib import Path             # tmp_path 夹具是 Path 类型

import pytest                        # 测试框架（参数化、tmp_path、raises）

from minimind_agent.agents.file_agent import _regex_extract, run_file_agent  # 文件专家：正则抽取 + 执行
from minimind_agent.config import AgentConfig                                # 配置
from minimind_agent.graph.nodes import _classify_fallback, _detect_agent, _result_source  # 兜底分类/检测/来源标注
from minimind_agent.graph.workflow import build_workflow                     # 整张图
from minimind_agent.provider import MockChatModel                           # 假模型
from minimind_agent.workspace import Workspace                              # 沙箱


# --------------------------------------------------------------------------- #
# Workspace 沙箱
# --------------------------------------------------------------------------- #
def test_workspace_move_into_dir(tmp_path: Path):  # 移动到目录：应进入目录并保留文件名
    ws = Workspace(tmp_path)         # 以临时目录建沙箱
    ws.seed_samples()                # 生成样例（含 inbox/note.txt）
    result = ws.move_file("inbox/note.txt", "archive/")  # 移动到 archive 目录
    assert result["ok"], result      # 应成功
    assert (tmp_path / "archive" / "note.txt").exists()  # 目标存在
    assert not (tmp_path / "inbox" / "note.txt").exists()  # 源已不在


def test_workspace_move_rejects_missing_source(tmp_path: Path):  # 源不存在应失败
    ws = Workspace(tmp_path)
    result = ws.move_file("inbox/does_not_exist.txt", "archive/")  # 移动不存在的文件
    assert result["ok"] is False     # 失败
    assert "不存在" in result["error"]  # 错误信息提示“不存在”


def test_workspace_move_refuses_overwrite(tmp_path: Path):  # 默认拒绝覆盖已存在目标
    ws = Workspace(tmp_path)
    ws.seed_samples()
    (tmp_path / "archive" / "note.txt").parent.mkdir(parents=True, exist_ok=True)  # 先造一个同名目标
    (tmp_path / "archive" / "note.txt").write_text("existing", encoding="utf-8")
    result = ws.move_file("inbox/note.txt", "archive/note.txt")  # 尝试覆盖
    assert result["ok"] is False     # 应被阻止
    assert "已存在" in result["error"]  # 提示“已存在”


def test_workspace_path_escape_blocked(tmp_path: Path):  # 越界路径应抛错（安全护栏）
    ws = Workspace(tmp_path)
    with pytest.raises(ValueError):  # 解析越界路径应抛 ValueError
        ws.resolve("../../secret.txt")


def test_workspace_syntax_check(tmp_path: Path):  # 语法校验：合法通过、非法报错
    ws = Workspace(tmp_path)
    ws.write_text("code/ok.py", "def f():\n    return 1\n")  # 合法代码
    ws.write_text("code/bad.py", "def f(:\n")                 # 故意写错
    assert ws.syntax_check("code/ok.py")["ok"] is True        # 合法 → 通过
    assert ws.syntax_check("code/bad.py")["ok"] is False       # 非法 → 不通过


# --------------------------------------------------------------------------- #
# file_agent 的路径抽取
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(            # 参数化：多种说法都应正确抽出“源/目标”
    "instruction,expected_src,expected_dst",
    [
        ("把 inbox/note.txt 移动到 archive 目录", "inbox/note.txt", "archive"),
        ("将 a.txt 移到 b/", "a.txt", "b/"),
        ("move inbox/data.csv to archive", "inbox/data.csv", "archive"),
        ("把 report.md 放到 docs 文件夹", "report.md", "docs"),
    ],
)
def test_file_regex_extract(instruction, expected_src, expected_dst):
    extracted = _regex_extract(instruction)  # 正则抽取
    assert extracted is not None     # 应能抽出
    src, dst, _is_dir = extracted    # 拆出源/目标
    assert src == expected_src       # 源符合预期
    assert dst == expected_dst       # 目标符合预期


def test_file_agent_moves_seeded_file(tmp_path: Path):  # file_agent 端到端：能移动样例文件
    ws = Workspace(tmp_path)
    ws.seed_samples()
    model = MockChatModel()          # 用假模型（其实移动不需要模型）
    step = {"instruction": "把 inbox/note.txt 移动到 archive 目录"}  # 构造一个步骤
    result = run_file_agent(model, ws, step)  # 执行
    assert result["ok"], result      # 成功
    assert (tmp_path / "archive" / "note.txt").exists()  # 文件已到 archive


# --------------------------------------------------------------------------- #
# 确定性兜底分类器
# --------------------------------------------------------------------------- #
def test_detect_agent_priority():    # 关键词检测优先级正确
    assert _detect_agent("把 a.txt 移动到 b") == "file"
    assert _detect_agent("写一个关于程序员的小故事") == "story"  # 不被“程序”误判成 code
    assert _detect_agent("写一个快速排序函数") == "code"


def test_classify_fallback_multi_step():  # 兜底分类：多子句任务切成正确的多步
    task = "写一个冒泡排序函数，然后写一个关于程序员的小故事，最后把 inbox/data.csv 移动到 archive"
    plan = _classify_fallback(task)  # 兜底规划
    agents = [step["agent"] for step in plan]  # 取每步的专家
    assert agents == ["code", "story", "file"]  # 顺序应为 代码→故事→文件


# --------------------------------------------------------------------------- #
# 端到端：mock 后端跑通整张图
# --------------------------------------------------------------------------- #
def _run(task: str, tmp_path: Path):  # 工具函数：用 mock 后端跑完整张图，返回最终状态
    cfg = AgentConfig.from_env(llm_mode="mock", workspace=tmp_path, seed_samples=True)  # mock 配置
    workflow = build_workflow()      # 构图
    return workflow.invoke({"task": task, "config": cfg, "messages": [], "max_replans": cfg.max_replans})  # 同步跑完


def test_e2e_code_task(tmp_path: Path):  # 端到端：写代码任务应产出可编译的 .py
    state = _run("写一个计算阶乘的Python函数", tmp_path)
    assert "final_answer" in state   # 有最终总结
    code_files = list((tmp_path / "code").glob("*.py"))  # 找产物
    assert code_files, "code_agent 应产出一个 .py 文件"
    ws = Workspace(tmp_path)
    assert ws.syntax_check(f"code/{code_files[0].name}")["ok"] is True  # 产物语法通过


def test_e2e_story_task(tmp_path: Path):  # 端到端：写故事任务应产出 .md
    state = _run("写一个关于小猫的睡前故事", tmp_path)
    assert "final_answer" in state
    assert list((tmp_path / "stories").glob("*.md")), "story_agent 应产出一个 .md 文件"


def test_e2e_file_task(tmp_path: Path):  # 端到端：移动文件任务应把文件挪到 archive
    state = _run("把 inbox/note.txt 移动到 archive 目录", tmp_path)
    assert (tmp_path / "archive" / "note.txt").exists()  # 文件已移动
    assert "[done]" in state["final_answer"]             # 总结里含完成标记
    assert all(step["status"] == "done" for step in state["plan"]), state["plan"]  # 所有步骤完成


# --------------------------------------------------------------------------- #
# 产物来源标注（区分“真实模型 / 兜底 / 正则”）
# --------------------------------------------------------------------------- #
def test_backend_label_flags_mock_and_local():  # 后端名标注正确（mock 必含“假模型”）
    assert "假模型" in AgentConfig.from_env(llm_mode="mock").backend_label()
    assert AgentConfig.from_env(llm_mode="local").backend_label() == "MiniMind(本地权重)"
    assert AgentConfig.from_env(llm_mode="server").backend_label() == "MiniMind(OpenAI服务)"


def test_result_source_tags():       # 来源标签：代码/故事区分模型vs兜底；文件区分正则vs模型抽取
    cfg = AgentConfig.from_env(llm_mode="local")
    assert "MiniMind(本地权重)" in _result_source(cfg, "code", {"used_fallback": False})  # 模型生成
    assert "兜底" in _result_source(cfg, "code", {"used_fallback": True})                 # 兜底
    assert "MiniMind(本地权重)" in _result_source(cfg, "story", {"used_fallback": False})
    assert "兜底" in _result_source(cfg, "story", {"used_fallback": True})
    assert "正则" in _result_source(cfg, "file", {"extracted_by": "regex"})               # 正则抽取
    assert "抽取" in _result_source(cfg, "file", {"extracted_by": "llm"})                  # 模型抽取


def test_e2e_source_is_recorded(tmp_path: Path):  # 端到端：每步都带 source，且 mock 标注为“假模型”
    cfg = AgentConfig.from_env(llm_mode="mock", workspace=tmp_path, seed_samples=True)
    workflow = build_workflow()
    state = workflow.invoke(
        {"task": "写一个关于小猫的故事", "config": cfg, "messages": [], "max_replans": cfg.max_replans}
    )
    assert all(step.get("source") for step in state["plan"]), state["plan"]  # 每步都有来源
    # mock 模式下，内容来源应明确标注为“假模型”，避免被误当成真实 MiniMind 输出
    assert any("假模型" in step.get("source", "") for step in state["plan"]), state["plan"]


def test_e2e_combined_task(tmp_path: Path):  # 端到端：组合任务三步全部完成、三类产物齐全
    task = "写一个冒泡排序函数，然后写一个关于程序员的小故事，最后把 inbox/data.csv 移动到 archive"
    state = _run(task, tmp_path)
    assert list((tmp_path / "code").glob("*.py"))     # 有代码产物
    assert list((tmp_path / "stories").glob("*.md"))  # 有故事产物
    assert (tmp_path / "archive" / "data.csv").exists()  # 文件已移动
    # 三步都应完成
    assert all(step["status"] == "done" for step in state["plan"]), state["plan"]
