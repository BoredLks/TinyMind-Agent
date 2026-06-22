# MiniMind Plan-and-Execute 多智能体

把本仓库里**本地部署的 MiniMind 大模型**当作 LLM 大脑，用 **LangGraph** 实现一个
“**先规划、再执行、可重试**”（Plan-and-Execute）的**多智能体**系统。

它参考了 **MokioAgent** 的实现思路（LangGraph 编排 + 多专家 Agent +
OpenAI 兼容的 LLM 接入层），并针对 MiniMind 这类约 64M 的超小模型做了大量稳健性处理。
系统包含三个专家 Agent：

| 专家 | 职责 | 实现要点 |
| --- | --- | --- |
| `file_agent` | 移动文件 | 确定性正则抽取“源/目标” + `shutil.move`，全程限制在 workspace 沙箱内 |
| `code_agent` | 编写简单 Python 代码 | LLM 生成 → 抽取代码块 → 落盘 → 内置 `compile()` 语法校验（默认只校验不执行） |
| `story_agent` | 编写小故事 | LLM 生成 → 落盘为 Markdown |

> 这是在原 MiniMind 本地化部署项目上的**增量扩展**：完全不改动原模型/训练代码，只新增
> `minimind_agent/` 包与 `run_agent.py` 入口，把模型“用起来”。

---

## 一、整体架构

外层 LangGraph 只有“规划 + 执行循环 + 复核重试”三块，三个专家以独立节点出现：

```text
                 ┌─────────────┐
   用户任务  ───▶ │   planner   │  MiniMind 产出 JSON 计划；解析失败则用确定性关键词分类兜底
                 └──────┬──────┘
                        │ route_after_step（取“第一个待办步骤”的 agent）
         ┌──────────────┼────────────────┐
         ▼              ▼                ▼
  ┌────────────┐  ┌────────────┐  ┌─────────────┐
  │ file_agent │  │ code_agent │  │ story_agent │   ← 三个专家（multi-agent）
  │  移动文件  │  │  写 Python │  │   写小故事  │
  └─────┬──────┘  └─────┬──────┘  └──────┬──────┘
        └───────────────┴── 执行循环 ─────┘  （每轮处理一个待办步骤，回到 route_after_step）
                        │ 没有待办步骤了
                        ▼
                  ┌──────────┐  有失败步骤且未超过 max_replans → 重置为待办，再跑一轮
                  │  replan  │───────────────────────────────────┐
                  └────┬─────┘                                   │
                       │ 无可重试 / 已达上限                       │ (重试)
                       ▼                                          │
                  ┌──────────┐                                    │
                  │  final   │ ◀──────────────────────────────────┘
                  └──────────┘  汇总：计划来源、各步结果、workspace 产物
```

- **planner**：让 MiniMind 输出 `[{"agent": "...", "instruction": "..."}]` 形式的计划。
  小模型对结构化 JSON 不可靠，因此解析失败时回退到**确定性关键词分类器**（按子句切分任务、
  逐句归到 file / code / story），保证一定能得到可执行计划。
- **三个专家节点**：每次取“第一个待办步骤”，按其 `agent` 分派给对应专家执行，更新步骤状态。
- **replan**：把失败步骤重置为待办再试一轮（受 `max_replans` 限制），避免一次失败就放弃。
- **final**：不调用模型，纯汇总成人类可读的报告。

## 二、为什么这么设计（针对超小模型的取舍）

MiniMind 约 64M，对“没见过的工具名/复杂 schema/长指令”的泛化能力有限。因此本项目**不依赖模型
稳定地吐出 tool_call**，而是“让模型做它擅长的、让 Python 做确定性的部分”：

- **规划**：模型尽力给 JSON，**确定性分类器兜底**。
- **移动文件**：源/目标优先用**正则确定性抽取**，真正的移动由 `shutil` 完成——稳定可复现。
- **写代码/写故事**：这是模型的强项（SFT 语料里就有“用 Python 写斐波那契”“写小故事”），
  让模型自由生成，再由程序负责落盘与语法校验。
- **稳健兜底**：即使 LLM 完全不可用（如 server 模式下没开服务），每个专家也会写出占位产物，
  整条流程不崩溃。

## 三、三种 LLM 后端

通过 `--mode` 或环境变量 `MINIMIND_LLM_MODE` 选择：

| 模式 | 说明 | 依赖 |
| --- | --- | --- |
| `local`（默认） | **进程内**直接加载 `out/full_sft_768.pth` 权重推理（复用 `eval_llm.py` 的推理配方） | 项目本身的 `torch` + `transformers` |
| `server` | 把 MiniMind 当成 OpenAI 兼容服务（先跑 `scripts/serve_openai_api.py`），用 `ChatOpenAI` 连接——与 MokioAgent 接入方式一致 | `langchain-openai` + `openai`（推理在另一个进程） |
| `mock` | 确定性假模型，无需 torch / 无需服务，用于跑通流程与单测 | 仅 `langgraph` + `langchain-core` |

## 四、安装

在**已能运行 MiniMind 的同一个 Python 环境**里安装额外依赖（这样 local 模式可直接用到已有的
torch / transformers）：

```bash
pip install -r requirements-agent.txt
```

> 只想用 `server` / `mock` 模式时，不需要 torch；`mock` 模式甚至不需要 `langchain-openai`。

## 五、运行示例

### local 模式（默认，进程内 MiniMind）

```bash
# 写代码
python run_agent.py "写一个快速排序的Python函数"

# 写故事
python run_agent.py "写一个关于小猫的睡前故事"

# 移动文件（首次运行会在 workspace 内自动生成 inbox/note.txt 等样例）
python run_agent.py "把 inbox/note.txt 移动到 archive 目录"

# 组合任务（一次规划出多步，依次交给不同专家）
python run_agent.py "写一个冒泡排序函数，然后写一个关于程序员的小故事，最后把 inbox/data.csv 移动到 archive"
```

### server 模式（复用 MiniMind 的 OpenAI 服务）

```bash
# 终端 1：启动 MiniMind 服务（仓库已有脚本）
cd scripts
python serve_openai_api.py

# 终端 2：让 Agent 连接它
python run_agent.py --mode server "写一个计算阶乘的Python函数"
```

### mock 模式（无需 torch / 无需服务，验证编排）

```bash
python run_agent.py --mode mock "写一个冒泡排序函数，然后把 inbox/note.txt 移动到 archive"
```

运行结束后，所有产物都在 workspace 目录（默认 `agent_workspace/`）：

```text
agent_workspace/
├─ inbox/            # 演示样例（note.txt / data.csv / readme.md）
├─ archive/          # 常见的“移动目标”目录
├─ code/             # code_agent 生成的 .py
└─ stories/          # story_agent 生成的 .md
```

### 可视化网页（Streamlit）

除了命令行，还提供一个把整个「规划 → 执行 → 复核」过程可视化的网页 [`agent_web.py`](agent_web.py)：

```bash
pip install -r requirements-agent.txt   # 需要 streamlit（已在 requirements.txt 内）
streamlit run agent_web.py
```

页面功能（覆盖了代码里的所有要素）：

- 左侧切换 **LLM 后端**（local / server / mock），mock 会醒目标注“假模型”；可调温度、top_p、最大生成长度、`max_replans`、是否允许执行代码、是否生成样例文件、local 权重/设备、server 地址等。
- 主区域输入任务（含一键示例），点击运行后**实时**显示：planner 计划（来源 llm/fallback）、每个专家步骤、每个产物的**来源徽章**（🟩MiniMind / 🟧兜底 / 🟦Python确定性 / 🟥假模型）、replan 重试、最终汇总。
- 运行概览指标（步数 / 成功 / 失败 / 重试轮）与**workspace 产物浏览器**（按目录分组查看 code/stories/archive，支持下载）。
- 侧边栏内含一张**工作流架构图**。

> 这是“给多智能体用”的页面；仓库自带的 `scripts/web_demo.py` 是“与原始模型聊天”的页面，二者用途不同。

## 六、输出里的「来源」标注（区分真实模型 vs 兜底）

运行时每一步都会直接标出产物的**真实来源**，最终总结里也逐条标注，避免把兜底/假数据误当成真实 MiniMind 输出：

- `内容由 MiniMind(本地权重) 生成` —— 代码/故事正文确实来自本地 MiniMind（`local` 模式）。
- `内容由 MiniMind(OpenAI服务) 生成` —— 来自 `server` 模式的 MiniMind 服务。
- `兜底(Python占位)` —— 模型没给出可用内容，写的是 Python 占位文本。
- `全程 Python(正则抽取源/目标 + shutil 移动)` —— 移动文件，未经过模型。
- `移动=Python(shutil)；源/目标由 ... 抽取` —— 移动是 Python，仅“源/目标”靠模型兜底抽取。

此外，一旦使用 `--mode mock`（假模型），启动时会**大声警告**，且后端名显示为 `MOCK假模型⚠(非真实模型)`，
所有产物来源都会带上这个标记 —— 假模型绝不会被伪装成真实输出。**默认就是 `local`（真实 MiniMind），不需要任何额外参数。**

## 七、常用参数

```bash
python run_agent.py --help
```

| 参数 | 说明 |
| --- | --- |
| `--mode {local,server,mock}` | 选择 LLM 后端 |
| `--workspace PATH` | 指定 workspace 沙箱根目录 |
| `--weight NAME` | local 模式权重前缀（默认 `full_sft`） |
| `--device {cuda,cpu}` | local 模式设备（默认自动） |
| `--base-url URL` | server 模式服务地址 |
| `--temperature FLOAT` | 生成温度 |
| `--max-replans N` | 失败后最多重试几轮（默认 1） |
| `--allow-exec` | 允许 `code_agent` **真正执行**生成的代码（默认仅语法校验） |
| `--no-seed` | 不生成演示样例文件 |

环境变量见 [`.env.agent.example`](.env.agent.example)。

## 八、安全说明

- **沙箱**：所有文件读写都被限制在 workspace 内（`Workspace.resolve` 做越界校验），不会动到工作区外的文件。
- **移动文件**：默认**拒绝覆盖**已存在的目标文件。
- **执行代码**：默认**只做语法校验，不执行**模型生成的代码。仅当显式加 `--allow-exec` 时，才会在子进程里
  带超时执行——请只在受控环境、对可信任务开启。

## 九、目录结构

```text
minimind_agent/
├─ __init__.py
├─ config.py            # AgentConfig：从 .env / 环境变量读取配置
├─ provider.py          # create_model()：local / server / mock 三后端工厂 + MockChatModel
├─ minimind_chat.py     # MiniMindChatModel：进程内封装 MiniMind 权重的 LangChain ChatModel
├─ prompts.py           # 各节点 / 专家的提示词
├─ parsing.py           # 从模型输出里容错抽取 JSON
├─ workspace.py         # Workspace 沙箱：安全路径、移动文件、写文件、语法校验、样例生成
├─ agents/
│  ├─ file_agent.py     # 移动文件（确定性正则 + shutil）
│  ├─ code_agent.py     # 写 Python（生成 → 落盘 → 语法校验）
│  └─ story_agent.py    # 写小故事（生成 → 落盘）
└─ graph/
   ├─ state.py          # PlanExecuteState / PlanStep
   ├─ nodes.py          # planner / 三专家节点 / 路由 / replan / final
   └─ workflow.py       # 装配并编译 StateGraph

run_agent.py            # 命令行入口
requirements-agent.txt  # 额外依赖
.env.agent.example      # 配置示例
tests_agent/            # 全程 mock 后端的单测 + 端到端测试
```

## 十、测试

```bash
python -m pytest tests_agent -q
```

测试全部使用 `mock` 后端，**不需要 torch、不需要启动服务**，覆盖：workspace 沙箱与越界拦截、
file_agent 的路径抽取、确定性兜底分类器、三类任务的端到端图运行、以及组合任务。

## 十一、已知限制

- MiniMind 很小，`local` / `server` 模式下生成的代码/故事质量有限，可能需要靠 `replan` 重试或人工微调；
  本项目保证的是**编排稳定、产物一定落盘、流程不崩溃**，而非模型本身的生成质量。
- `planner` 的 LLM JSON 经常不规范，多数情况下会落到**确定性兜底分类器**——这本身就是面向小模型的设计。
- 复杂的多文件移动、跨目录批量操作不在演示范围内（聚焦三类清晰任务）。
```
