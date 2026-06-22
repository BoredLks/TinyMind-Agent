"""agent_web.py —— MiniMind Plan-and-Execute 多智能体的 Streamlit 可视化网页（深色科技风）。

把 minimind_agent（planner → file/code/story 三专家 → replan → final）的整个运行过程可视化：
配置 LLM 后端、输入任务、实时看到“规划 / 每步执行 / 每个产物的真实来源 / 重试 / 最终汇总”，
并能浏览、下载 workspace 沙箱里生成/移动的文件。

运行：
    streamlit run agent_web.py

注意：这是“给多智能体用”的页面，和仓库里 scripts/web_demo.py（与原始模型聊天）用途不同。
视觉主题由 .streamlit/config.toml（暗色基底）+ 本文件注入的 CSS（霓虹/玻璃拟态）共同决定。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 把项目根加入搜索路径，保证能 import minimind_agent（本文件就在项目根）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st  # 网页框架

st.set_page_config(  # 页面基础设置（必须是第一个 st 调用）
    page_title="MiniMind 多智能体",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- 惰性导入 agent 依赖：缺 langgraph 等依赖时给出友好提示，而不是整页崩溃 ----
try:
    from minimind_agent.config import AgentConfig, PROJECT_ROOT, VALID_LLM_MODES  # 配置
    from minimind_agent.graph.workflow import build_workflow  # 编译好的图
    from minimind_agent.workspace import Workspace  # 沙箱（浏览产物）
    _IMPORT_ERROR = None
except Exception as exc:  # 依赖缺失/导入异常
    _IMPORT_ERROR = exc

# ===========================================================================
# 科技风样式：霓虹渐变 + 玻璃拟态 + 辉光背景 + Orbitron 字体（整段为静态 CSS）
# ===========================================================================
TECH_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800;900&family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root{
  --cyan:#22d3ee; --violet:#a855f7; --indigo:#6366f1;
  --grad:linear-gradient(100deg,#22d3ee 0%,#6366f1 50%,#a855f7 100%);
  --txt:#e6ecf5; --muted:#8a97ad; --panel:rgba(146,180,255,.05); --brd:rgba(120,180,255,.16);
  --ok:#34d399; --bad:#fb7185; --warn:#fbbf24;
}

/* ===== 背景：深色 + 双径向辉光 + 顶部细网格 ===== */
.stApp{
  background:
    radial-gradient(1000px 520px at 10% -10%, rgba(34,211,238,.13), transparent 60%),
    radial-gradient(1000px 620px at 100% -5%, rgba(168,85,247,.13), transparent 55%),
    linear-gradient(180deg,#070b16 0%,#05080f 100%) !important;
  color:var(--txt);
  font-family:'Rajdhani','Inter','Microsoft YaHei',sans-serif;
}
.stApp::before{ /* 细网格叠加，增强科技感 */
  content:""; position:fixed; inset:0; pointer-events:none; opacity:.35; z-index:0;
  background-image:linear-gradient(rgba(120,180,255,.05) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(120,180,255,.05) 1px,transparent 1px);
  background-size:42px 42px; mask-image:linear-gradient(180deg,#000,transparent 70%);
}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stToolbar"]{right:1rem;}
.block-container{padding-top:2.2rem; position:relative; z-index:1;}

h1,h2,h3,h4{font-family:'Orbitron','Rajdhani',sans-serif; letter-spacing:.4px; color:var(--txt);}

/* ===== Hero 头部 ===== */
.mm-hero{position:relative; overflow:hidden; padding:26px 30px; border-radius:20px; margin-bottom:14px;
  background:linear-gradient(135deg,rgba(34,211,238,.10),rgba(168,85,247,.10));
  border:1px solid var(--brd);
  box-shadow:0 0 0 1px rgba(255,255,255,.02), 0 24px 60px -28px rgba(99,102,241,.55);}
.mm-hero::after{content:""; position:absolute; top:0; left:-60%; width:60%; height:100%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.07),transparent); animation:mm-sheen 7s linear infinite;}
@keyframes mm-sheen{0%{left:-60%}100%{left:160%}}
.mm-kicker{font-family:'JetBrains Mono',monospace; font-size:12px; letter-spacing:4px; color:var(--cyan);
  text-transform:uppercase; margin:0 0 4px;}
.mm-title{font-family:'Orbitron'; font-weight:900; font-size:clamp(26px,4vw,44px); line-height:1.04; margin:0 0 6px;
  background:var(--grad); -webkit-background-clip:text; background-clip:text; color:transparent;
  filter:drop-shadow(0 0 22px rgba(99,102,241,.5));}
.mm-sub{color:var(--muted); font-size:15px; margin:0;}
.mm-dot{display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--cyan);
  box-shadow:0 0 12px var(--cyan); margin-right:7px; animation:mm-pulse 1.6s infinite;}
@keyframes mm-pulse{0%,100%{opacity:1}50%{opacity:.25}}

/* 状态 chip */
.mm-chip{display:inline-flex; align-items:center; gap:7px; padding:5px 13px; border-radius:30px;
  font-family:'JetBrains Mono',monospace; font-size:12.5px; border:1px solid var(--brd);
  background:rgba(255,255,255,.04); margin-right:10px;}
.mm-chip b{color:var(--txt);}

/* ===== 按钮 ===== */
.stButton>button{
  background:rgba(255,255,255,.035); color:var(--txt); border:1px solid var(--brd); border-radius:12px;
  font-family:'Rajdhani',sans-serif; font-weight:600; letter-spacing:.6px; transition:all .18s ease;}
.stButton>button:hover{border-color:var(--cyan); color:var(--cyan); transform:translateY(-1px);
  box-shadow:0 0 22px -6px var(--cyan);}
.stButton>button[kind="primary"], [data-testid="stBaseButton-primary"]{
  background:var(--grad)!important; color:#04070e!important; border:none!important; font-weight:800!important;
  text-transform:uppercase; letter-spacing:1.2px; box-shadow:0 10px 34px -10px rgba(99,102,241,.75)!important;}
.stButton>button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover{
  filter:brightness(1.08); box-shadow:0 12px 44px -8px rgba(34,211,238,.85)!important;}

/* ===== 侧栏 ===== */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0a1020,#070b16); border-right:1px solid var(--brd);}
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:var(--txt);}

/* ===== 输入 / 选择 / 文本域 ===== */
.stTextInput input,.stTextArea textarea,.stNumberInput input,
[data-baseweb="input"],[data-baseweb="textarea"],[data-baseweb="select"]>div{
  background:rgba(255,255,255,.04)!important; border-color:var(--brd)!important; color:var(--txt)!important;
  border-radius:10px!important; font-family:'JetBrains Mono',monospace;}
.stTextArea textarea:focus,.stTextInput input:focus{box-shadow:0 0 0 1px var(--cyan)!important; border-color:var(--cyan)!important;}

/* ===== 指标卡 ===== */
[data-testid="stMetric"],[data-testid="metric-container"]{
  background:var(--panel); border:1px solid var(--brd); border-radius:14px; padding:12px 14px;
  box-shadow:inset 0 0 36px -26px var(--cyan);}
[data-testid="stMetricValue"]{font-family:'Orbitron'; color:var(--cyan); text-shadow:0 0 16px rgba(34,211,238,.55);}
[data-testid="stMetricLabel"]{color:var(--muted)!important; text-transform:uppercase; letter-spacing:1px; font-size:11px!important;}

/* ===== expander / 代码块 ===== */
[data-testid="stExpander"]{background:var(--panel); border:1px solid var(--brd)!important; border-radius:14px;}
details summary{color:var(--txt)!important;}
[data-testid="stCodeBlock"],pre{border:1px solid var(--brd); border-radius:12px; box-shadow:0 0 50px -32px var(--cyan);}
code,kbd,pre{font-family:'JetBrains Mono',monospace!important;}

/* ===== 提示框（status/alert）微调 ===== */
[data-testid="stNotification"],[data-testid="stStatusWidget"]{border:1px solid var(--brd); border-radius:12px;}

/* ===== 自定义卡片 / 步骤 / 徽章 ===== */
.mm-card{background:var(--panel); border:1px solid var(--brd); border-radius:14px; padding:14px 18px; margin:10px 0; backdrop-filter:blur(8px);}
.mm-plan-h{font-family:'Orbitron'; font-size:15px; color:var(--txt); margin-bottom:8px;}
.mm-step{position:relative; border-radius:12px; padding:11px 15px; margin:9px 0;
  background:rgba(255,255,255,.035); border:1px solid var(--brd); border-left:3px solid var(--cyan); transition:all .15s ease;}
.mm-step:hover{background:rgba(255,255,255,.06);}
.mm-step.file{border-left-color:#34d399;} .mm-step.code{border-left-color:#22d3ee;} .mm-step.story{border-left-color:#a855f7;}
.mm-step b{color:var(--txt);}
.mm-instr{color:var(--muted); font-size:13px; font-family:'JetBrains Mono',monospace;}
.mm-badge{display:inline-block; padding:2px 11px; border-radius:30px; font-size:11.5px; font-weight:700;
  font-family:'JetBrains Mono',monospace; border:1px solid; vertical-align:middle; margin-left:6px;}
.mm-num{font-family:'Orbitron'; color:var(--cyan); margin-right:6px;}

/* ===== 滚动条 ===== */
::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-thumb{background:linear-gradient(180deg,var(--cyan),var(--violet)); border-radius:10px}
::-webkit-scrollbar-track{background:transparent}
hr{border-color:var(--brd);}
</style>
"""
st.markdown(TECH_CSS, unsafe_allow_html=True)

# 三个专家的展示信息：emoji + 中文名 + css 类（决定左侧色条颜色）
AGENT_META = {
    "file": ("📁", "file_agent · 移动文件", "file"),
    "code": ("💻", "code_agent · 写 Python", "code"),
    "story": ("📖", "story_agent · 写故事", "story"),
}


def source_style(source: str) -> tuple[str, str, str]:
    """把“产物来源”文本映射成（emoji, 简短标签, 颜色），用于徽章。"""
    if "假模型" in source:  # MOCK 模式
        return ("◆", "MOCK 假模型", "#fb7185")
    if "兜底" in source:  # Python 占位兜底
        return ("◆", "兜底占位", "#fbbf24")
    if "MiniMind" in source:  # 真实模型生成
        return ("◆", "MiniMind 生成", "#34d399")
    return ("◆", "Python 确定性", "#22d3ee")  # 正则/shutil 等


def badge_html(text: str, color: str) -> str:
    """生成一个霓虹发光的胶囊徽章 HTML（半透明填充 + 同色边框/辉光）。"""
    return (
        f'<span class="mm-badge" style="color:{color};border-color:{color};'
        f'background:{color}1f;box-shadow:0 0 14px -4px {color};">{text}</span>'
    )


def chip_html(label: str, value: str, color: str = "#22d3ee") -> str:
    """状态 chip：左侧一个发光圆点 + 标签:值。"""
    return (
        f'<span class="mm-chip"><span style="width:8px;height:8px;border-radius:50%;'
        f'background:{color};box-shadow:0 0 10px {color};display:inline-block;"></span>'
        f'{label} <b>{value}</b></span>'
    )


# ===========================================================================
# 架构图（DOT）：浏览器端 viz.js 渲染，无需本机装 graphviz；配色贴合暗色主题
# ===========================================================================
WORKFLOW_DOT = """
digraph G {
  rankdir=TB; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fontname="Rajdhani", fontsize=11, color="#2a3550", fontcolor="#e6ecf5"];
  edge [fontname="Rajdhani", fontsize=10, color="#5a6a90", fontcolor="#8a97ad"];
  START [shape=circle, label="开始", fillcolor="#13203a"];
  planner [label="planner 规划\\n(MiniMind 出 JSON 计划 / 失败则确定性兜底)", fillcolor="#1a2342"];
  file_agent [label="file_agent\\n移动文件", fillcolor="#103027"];
  code_agent [label="code_agent\\n写 Python", fillcolor="#0c2d39"];
  story_agent [label="story_agent\\n写小故事", fillcolor="#241433"];
  replan [label="replan\\n复核·失败重试", fillcolor="#3a1622"];
  final [label="final 汇总", fillcolor="#13243f"];
  END [shape=circle, label="结束", fillcolor="#13203a"];
  START -> planner;
  planner -> file_agent; planner -> code_agent; planner -> story_agent;
  file_agent -> replan; code_agent -> replan; story_agent -> replan;
  replan -> file_agent [label="重试", style=dashed];
  replan -> final;
  final -> END;
}
"""


# ===========================================================================
# 事件渲染：把图运行期产生的每个自定义事件画到页面上
# ===========================================================================
def render_event(container, event: dict) -> None:
    """渲染单个事件（plan / step_start / step_result / replan / planner_warning）。"""
    etype = event.get("type")        # 事件类型
    if etype == "plan":              # 计划事件：渲染来源 + 各步卡片
        source = event.get("source")  # llm / fallback
        backend = event.get("backend", "")  # 后端名
        plan_from = f"{backend} 规划" if source == "llm" else "确定性关键词分类器(Python 兜底)"  # 计划来源描述
        container.markdown(f'<div class="mm-plan-h">🧭 planner 生成计划 · 来源：{plan_from}</div>', unsafe_allow_html=True)  # 计划标题
        for i, step in enumerate(event.get("plan", []), 1):  # 逐步渲染
            emoji, name, cls = AGENT_META.get(step.get("agent", ""), ("•", step.get("agent", ""), ""))  # 取专家展示信息
            container.markdown(      # 渲染一张步骤卡（左侧色条按专家着色）
                f'<div class="mm-step {cls}"><span class="mm-num">{i:02d}</span>{emoji} <b>{name}</b>'
                f'<br><span class="mm-instr">{step.get("instruction", "")}</span></div>',
                unsafe_allow_html=True,
            )
    elif etype == "planner_warning":  # 规划告警 → info 框
        container.info(f"planner 提示：{event.get('message', '')}")
    elif etype == "step_start":      # 某步开始 → 一行“执行中”
        emoji, name, cls = AGENT_META.get(event.get("agent", ""), ("•", event.get("agent", ""), ""))
        container.markdown(f'<div class="mm-instr">▶ {emoji} <b>{name}</b> 执行中…</div>', unsafe_allow_html=True)
    elif etype == "step_result":     # 某步结果 → 卡片 + 来源徽章
        emoji, name, cls = AGENT_META.get(event.get("agent", ""), ("•", event.get("agent", ""), ""))
        ok = bool(event.get("ok"))   # 是否成功
        mark = "✅" if ok else "❌"   # 成功/失败标记
        s_emoji, s_label, s_color = source_style(str(event.get("source", "")))  # 来源 → 徽章样式
        container.markdown(          # 渲染结果卡（含彩色来源徽章）
            f'<div class="mm-step {cls}">{mark} {emoji} <b>{name}</b> '
            f'{badge_html(s_emoji + " " + s_label, s_color)}'
            f'<br><span class="mm-instr">→ {event.get("result", "")}</span></div>',
            unsafe_allow_html=True,
        )
    elif etype == "replan":          # 重试事件 → warning 框
        container.warning(f"↻ replan 第 {event.get('replans')} 次重试，重置失败步骤：{event.get('retry')}")


def render_events(container, events: list[dict]) -> None:
    """按顺序渲染一组事件（用于运行后/重跑后的持久展示）。"""
    for event in events:
        render_event(container, event)


def render_summary(state: dict) -> None:
    """渲染运行概览：后端、指标、各步来源一览。"""
    events = state.get("events", [])  # 本次运行收集到的所有事件
    step_results = [e for e in events if e.get("type") == "step_result"]  # 只取“步骤结果”事件
    plan_event = next((e for e in events if e.get("type") == "plan"), None)  # 找“计划”事件
    total = len(step_results)        # 总步数
    done = sum(1 for e in step_results if e.get("ok"))  # 成功步数
    failed = total - done            # 失败步数
    replans = max((e.get("replans", 0) for e in events if e.get("type") == "replan"), default=0)  # 重试轮数

    st.markdown("### 📊 运行概览")
    cols = st.columns(5)             # 5 个指标卡并排
    cols[0].metric("LLM 后端", state.get("backend", "—"))
    cols[1].metric("计划步数", total)
    cols[2].metric("成功", done)
    cols[3].metric("失败", failed)
    cols[4].metric("重试轮", replans)

    if plan_event is not None:
        src = plan_event.get("source")
        st.caption(
            "计划来源："
            + (f"{plan_event.get('backend','')}（模型规划）" if src == "llm" else "确定性关键词分类器（Python 兜底）")
        )

    # 各步来源一览（哪一步是 MiniMind 写的、哪一步是兜底/确定性）
    if step_results:
        st.markdown("**各步产物来源：**")
        for i, e in enumerate(step_results, 1):  # 逐步列出 + 来源徽章
            emoji, name, cls = AGENT_META.get(e.get("agent", ""), ("•", e.get("agent", ""), ""))  # 专家信息
            s_emoji, s_label, s_color = source_style(str(e.get("source", "")))  # 来源样式
            mark = "✅" if e.get("ok") else "❌"  # 成功/失败标记
            st.markdown(
                f'{mark} <span class="mm-num">{i:02d}</span> {emoji} {name} '
                f'{badge_html(s_emoji + " " + s_label, s_color)} '
                f'<span class="mm-instr">{e.get("artifact","")}</span>',
                unsafe_allow_html=True,
            )


def render_workspace(ws_path: str) -> None:
    """浏览 workspace 沙箱里的所有文件，支持查看与下载。"""
    st.markdown("### 📂 workspace 产物")
    if _IMPORT_ERROR is not None:    # 依赖缺失时无法读取，直接返回
        return
    try:
        ws = Workspace(Path(ws_path))  # 以该路径建沙箱
        files = ws.list_files()        # 列出所有文件
    except Exception as exc:  # 路径异常等
        st.warning(f"无法读取 workspace：{exc}")
        return
    st.caption(f"沙箱目录：{ws_path}")
    if not files:                    # 没文件
        st.info("workspace 暂无文件。")
        return

    # 按一级目录分组展示（inbox / archive / code / stories / 其它）
    groups: dict[str, list[str]] = {}  # 一级目录 → 文件列表
    for rel in files:
        top = rel.replace("\\", "/").split("/")[0] if ("/" in rel or "\\" in rel) else "(根目录)"  # 取一级目录名
        groups.setdefault(top, []).append(rel)

    for top in sorted(groups):       # 每个目录一个折叠面板
        with st.expander(f"📁 {top} · {len(groups[top])} 个文件", expanded=(top in ("code", "stories", "archive"))):
            for rel in sorted(groups[top]):  # 逐个文件展示
                read = ws.read_text(rel)     # 读内容
                content = read.get("content", "") if read.get("ok") else f"(无法读取：{read.get('error','')})"
                st.markdown(f"**{rel}**")    # 文件名
                lower = rel.lower()          # 按扩展名选高亮
                if lower.endswith(".py"):
                    st.code(content, language="python")  # 代码高亮
                elif lower.endswith(".md"):
                    st.markdown(content)     # Markdown 渲染
                elif lower.endswith((".json",)):
                    st.code(content, language="json")
                else:
                    st.code(content, language="text")
                st.download_button(          # 提供下载按钮
                    "⬇ 下载",
                    data=content.encode("utf-8"),
                    file_name=os.path.basename(rel),
                    key=f"dl_{rel}",
                )


# ===========================================================================
# 侧边栏：依赖检查 + 后端与参数配置 + 架构说明
# ===========================================================================
with st.sidebar:
    st.markdown("## ⚙️ 控制台")

    if _IMPORT_ERROR is not None:  # 缺依赖：给出安装提示
        st.error(f"无法导入 minimind_agent：{type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}")
        st.code("pip install -r requirements-agent.txt", language="bash")

    mode = st.radio(
        "LLM 后端",
        options=["local", "server", "mock"],
        format_func=lambda m: {
            "local": "local · 进程内 MiniMind 权重",
            "server": "server · 连 serve_openai_api.py",
            "mock": "mock · 假模型(测试用)",
        }[m],
        key="mode",
        help="local 需要 torch；server 需先启动 serve_openai_api.py；mock 不需要模型，仅验证流程",
    )
    if mode == "mock":
        st.warning("当前是 **mock 假模型**：所有产物均为占位假数据，不是真实 MiniMind 输出。")

    default_ws = str(PROJECT_ROOT / "agent_workspace") if _IMPORT_ERROR is None else "agent_workspace"
    workspace = st.text_input("workspace 目录（文件操作沙箱）", value=default_ws, key="workspace")
    seed_samples = st.checkbox("在 workspace 生成演示样例文件（inbox/...）", value=True, key="seed_samples",
                               help="便于直接体验“移动文件”：会生成 inbox/note.txt 等")

    with st.expander("生成参数 / 高级", expanded=False):
        temperature = st.slider("温度 temperature", 0.0, 1.5, 0.6, 0.05, key="temperature")
        top_p = st.slider("top_p", 0.1, 1.0, 0.9, 0.05, key="top_p")
        max_new_tokens = st.slider("最大生成长度 max_new_tokens", 128, 4096, 1024, 128, key="max_new_tokens")
        max_replans = st.slider("失败后最多重试轮数 max_replans", 0, 3, 1, 1, key="max_replans")
        allow_exec = st.checkbox("允许 code_agent 真正执行生成的代码（默认仅语法校验）", value=False, key="allow_exec")
        if allow_exec:
            st.warning("已开启代码执行：会在子进程里运行模型生成的代码，请仅在可信任务上使用。")

    with st.expander("local 模式（进程内权重）", expanded=(mode == "local")):
        weight = st.text_input("权重前缀 weight", value="full_sft", key="weight")
        hidden_size = st.number_input("hidden_size", value=768, step=64, key="hidden_size")
        device = st.text_input("device（留空=自动）", value="", key="device")
        st.caption("权重文件应位于 out/{weight}_{hidden_size}.pth，并需要 torch + transformers。")

    with st.expander("server 模式（OpenAI 兼容服务）", expanded=(mode == "server")):
        base_url = st.text_input("base_url", value="http://127.0.0.1:8998/v1", key="base_url")
        model_name = st.text_input("model 名称", value="minimind", key="model_name")
        st.caption("需先运行： cd scripts && python serve_openai_api.py")

    with st.expander("🧭 工作流架构", expanded=False):
        st.graphviz_chart(WORKFLOW_DOT, use_container_width=True)
        st.caption("外层只有“规划 + 执行循环 + 复核重试”，三个专家以独立节点出现。")


# ===========================================================================
# 主区域：Hero 头部 + 状态 chip + 任务输入 + 运行 + 可视化
# ===========================================================================
_mode = st.session_state.get("mode", "local")
_mode_color = "#fb7185" if _mode == "mock" else "#22d3ee"
_ws_now = st.session_state.get("workspace", "")

st.markdown(
    f"""
    <div class="mm-hero">
      <div class="mm-kicker">◇ MINIMIND · MULTI-AGENT</div>
      <div class="mm-title">PLAN · EXECUTE AGENT</div>
      <p class="mm-sub"><span class="mm-dot"></span>本地 MiniMind 驱动的多智能体 —— 规划 → 执行 → 复核，把“移动文件 / 写 Python / 写故事”交给三个专家。</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    chip_html("后端", _mode.upper(), _mode_color)
    + chip_html("沙箱", os.path.basename(str(_ws_now).rstrip("/\\")) or "—", "#a855f7"),
    unsafe_allow_html=True,
)

# 任务输入框默认值（首次进入）
if "task_input" not in st.session_state:
    st.session_state["task_input"] = "写一个快速排序的Python函数，然后写一个关于程序员的小故事"

# 例子按钮（必须在 text_area 之前设置 session_state）
st.markdown("###### ⚡ 快速示例")
ex_cols = st.columns(4)
_examples = [
    ("✍️ 写代码", "写一个计算阶乘的Python函数"),
    ("📖 写故事", "写一个关于小猫的睡前故事"),
    ("📁 移动文件", "把 inbox/note.txt 移动到 archive 目录"),
    ("🔗 组合任务", "写一个冒泡排序函数，然后写一个关于程序员的小故事，最后把 inbox/data.csv 移动到 archive"),
]
for col, (label, text) in zip(ex_cols, _examples):
    if col.button(label, use_container_width=True, key=f"ex_{label}"):
        st.session_state["task_input"] = text

task = st.text_area("任务描述", key="task_input", height=90,
                    placeholder="例如：写一个二分查找的Python函数")

run_clicked = st.button("🚀 运行智能体", type="primary", key="run_btn", use_container_width=True,
                        disabled=(_IMPORT_ERROR is not None or not str(task).strip()))


def build_config():
    """根据侧边栏配置组装 AgentConfig（只传非空 override）。"""
    overrides = dict(
        llm_mode=st.session_state.get("mode"),
        workspace=st.session_state.get("workspace") or None,
        seed_samples=st.session_state.get("seed_samples"),
        temperature=st.session_state.get("temperature"),
        top_p=st.session_state.get("top_p"),
        max_new_tokens=st.session_state.get("max_new_tokens"),
        max_replans=st.session_state.get("max_replans"),
        allow_exec=st.session_state.get("allow_exec"),
        weight=st.session_state.get("weight") or None,
        hidden_size=int(st.session_state.get("hidden_size") or 768),
        device=(st.session_state.get("device") or None),
        base_url=st.session_state.get("base_url") or None,
        model_name=st.session_state.get("model_name") or None,
    )
    overrides = {k: v for k, v in overrides.items() if v is not None}  # 过滤掉 None，只覆盖显式设置的项
    return AgentConfig.from_env(**overrides)  # 叠加到环境默认配置上


if run_clicked:                      # 点击了“运行智能体”
    cfg = build_config()             # 组装配置
    st.divider()
    st.markdown("### 🔄 运行过程")
    log = st.container()  # 实时事件流容器
    events: list[dict] = []          # 收集事件（供概览/持久展示）
    final_answer = ""                # 最终总结
    ok = True                        # 运行是否成功
    with st.status("智能体运行中…（local 模式首次会加载模型，可能较慢）", expanded=True) as status:  # 运行状态条
        try:
            workflow = build_workflow()  # 构建图
            inputs = {"task": task, "config": cfg, "messages": [], "max_replans": cfg.max_replans}  # 初始状态
            for stream_mode, payload in workflow.stream(inputs, stream_mode=["updates", "custom"]):  # 双流运行
                if stream_mode == "custom":  # 自定义事件 → 实时画出来
                    events.append(payload)
                    render_event(log, payload)
                elif isinstance(payload, dict):  # 图更新 → 捕获最终答案
                    for update in payload.values():
                        if isinstance(update, dict) and update.get("final_answer"):
                            final_answer = update["final_answer"]
            status.update(label="运行完成 ✅", state="complete")  # 标记完成
        except Exception as exc:  # 任何异常都不让页面崩溃
            ok = False
            status.update(label=f"运行出错：{type(exc).__name__}", state="error")  # 标记出错
            st.exception(exc)        # 显示异常栈

    # 把本次运行结果存入 session_state，便于后续 rerun（点开文件/下载）时仍能展示
    st.session_state["last_run"] = {
        "events": events,
        "final_answer": final_answer,
        "workspace": str(cfg.workspace),
        "backend": cfg.backend_label(),
        "task": task,
        "ok": ok,
    }

    st.divider()
    render_summary(st.session_state["last_run"])
    if final_answer:
        st.markdown("### 🏁 最终汇总")
        st.code(final_answer, language="text")
    render_workspace(str(cfg.workspace))

elif st.session_state.get("last_run"):
    # 非“刚运行”的 rerun（例如点了下载按钮）：从 session_state 重新渲染上一次运行
    state = st.session_state["last_run"]
    st.divider()
    st.markdown(f"### 🔄 上次运行过程 · 任务：{state.get('task','')}")
    container = st.container()
    render_events(container, state.get("events", []))
    st.divider()
    render_summary(state)
    if state.get("final_answer"):
        st.markdown("### 🏁 最终汇总")
        st.code(state["final_answer"], language="text")
    render_workspace(state.get("workspace", ""))

else:
    # 首次进入的引导
    st.info("在上方输入任务并点击「🚀 运行智能体」。左侧可切换 LLM 后端与参数；mock 后端无需模型即可体验完整流程。")
