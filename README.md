<div align="center">

# 🧠 TinyMind

**从零训练一个超小语言模型，只需 3 块钱、2 小时**

![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c)
![License](https://img.shields.io/badge/license-Apache%202.0-green)

*"大道至简"——用最简洁的代码，从零训练一个能对话的语言模型*

</div>

---

## 📌 项目简介

TinyMind 是一个完全从零开始训练超小语言模型（~64M 参数）的开源项目。

项目覆盖了从 Tokenizer 训练、预训练（Pretrain）、监督微调（SFT）、LoRA、强化学习（DPO / PPO / GRPO / CISPO）、Agent RL 到知识蒸馏的**全流程代码**，所有核心算法均使用 PyTorch 原生实现，不依赖第三方高层抽象接口。

此外，项目还包含一个基于 LangGraph 的 **Plan-and-Execute 多智能体系统**，展示了如何将小型 LLM 作为 Agent 大脑来执行多步骤任务。

> **目标**：让每个人都能从理解每一行代码开始，亲手训练一个语言模型。

---

## ✨ 核心特性

<table>
<tr>
<td width="50%">

### 🏗️ 模型训练

- Transformer Decoder-Only 结构，对齐 Qwen3 生态
- 完整训练链路：Pretrain → SFT → LoRA → DPO → PPO → GRPO → CISPO → Agent RL
- 所有核心算法纯 PyTorch 原生实现
- 支持单机单卡 / 多卡（DDP / DeepSpeed）
- 支持 wandb / swanlab 训练可视化
- 支持断点续训

</td>
<td width="50%">

### 🤖 Agent 多智能体

- 基于 LangGraph 的 Plan-and-Execute 架构
- 三个专家 Agent：文件操作 / 代码生成 / 故事编写
- 支持 local / server / mock 三种 LLM 后端
- Streamlit 可视化 WebUI
- 自动生成 workspace 产物

</td>
</tr>
<tr>
<td>

### 🛠️ 工具调用 & 思考

- 原生 Tool Calling 能力（已混入 SFT 数据）
- 自适应思考（`<think>` 标签 + `open_thinking` 开关）
- 兼容 OpenAI API 协议
- 支持 `reasoning_content` / `tool_calls`

</td>
<td>

### 🔌 多框架兼容

- transformers / llama.cpp / vllm / ollama
- Streamlit WebUI 聊天界面
- OpenAI 兼容 API 服务端
- 模型格式互转（torch ↔ transformers）

</td>
</tr>
</table>

---

## 🚀 快速开始

### 前置要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | 3.10+ | 推荐 3.10 |
| PyTorch | 2.0+ | 建议 CUDA 11.8+ |
| GPU | NVIDIA 3090 (24GB) 推荐 | CPU 也可运行但较慢 |

### 1. 获取代码

```bash
git clone https://github.com/BoredLks/TinyMind-Agent.git
cd TinyMind-Agent
```

或直接下载 ZIP 源码并解压。

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载模型

```bash
# 方式 1：从 ModelScope 下载
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3

# 方式 2：从 HuggingFace 下载
git clone https://huggingface.co/jingyaogong/minimind-3
```

### 4. 开始对话

```bash
# 使用 Transformers 格式模型（推荐）
python eval_llm.py --load_from ./minimind-3

# 使用 PyTorch 权重（需先训练或下载 .pth 文件放入 ./out/ 目录）
python eval_llm.py --load_from ./model --weight full_sft
```

> **注意**：`./out/` 目录是训练产物输出目录，不包含在仓库中。运行训练脚本后会自动创建，或从 ModelScope 下载 `.pth` 权重文件放入 `./out/` 目录。

<!-- screenshot: cli_demo -->
<!-- 截图文件名：images/cli_demo.png -->
<!-- 截图说明：终端运行 eval_llm.py 后的多轮对话截图 -->

### 5. 启动 WebUI（可选）

```bash
# 将模型文件夹复制到 scripts 目录下
cp -r minimind-3 ./scripts/minimind-3
cd scripts && streamlit run web_demo.py
```

<!-- screenshot: webui_demo -->
<!-- 截图文件名：images/webui_demo.png -->
<!-- 截图说明：Streamlit WebUI 聊天界面截图 -->

---

## 🛠️ 模型训练

### 数据准备

从 [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) 或 [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main) 下载数据集，放入 `./dataset/` 目录。

> 无需全部下载，可单独下载所需文件。最小只需 `pretrain_t2t_mini.jsonl` + `sft_t2t_mini.jsonl` 即可快速复现。

| 文件 | 大小 | 说明 | 推荐 |
|------|------|------|------|
| `pretrain_t2t_mini.jsonl` | 1.2GB | 轻量预训练数据 | ✨ 快速复现 |
| `sft_t2t_mini.jsonl` | 1.6GB | 轻量 SFT 数据（含 Tool Call） | ✨ 快速训练 |
| `pretrain_t2t.jsonl` | 10GB | 完整预训练数据 | 完整训练 |
| `sft_t2t.jsonl` | 14GB | 完整 SFT 数据 | 完整训练 |
| `rlaif.jsonl` | 24MB | RLAIF 训练数据 | ✨ 强化学习 |
| `dpo.jsonl` | 53MB | DPO 偏好数据 | 可选 |
| `agent_rl.jsonl` | 86MB | Agentic RL 数据 | 可选 |
| `agent_rl_math.jsonl` | 18MB | Agent 数学补充数据 | 可选 |

### 训练流程

> 所有训练脚本均在 `./trainer` 目录下执行

#### Step 1：预训练（必须）

```bash
cd trainer
python train_pretrain.py
# 多卡训练：torchrun --nproc_per_node N train_pretrain.py
```

<!-- screenshot: pretrain_loss -->
<!-- 截图文件名：images/pretrain_loss.png -->
<!-- 截图说明：预训练阶段的 loss 曲线图 -->

> 训练后得到 `out/pretrain_*.pth` 权重文件

#### Step 2：监督微调（必须）

```bash
python train_full_sft.py
```

> 训练后得到 `out/full_sft_*.pth` 权重文件

#### Step 3：强化学习（可选）

```bash
# DPO 偏好对齐
python train_dpo.py

# PPO 强化学习
python train_ppo.py

# GRPO / CISPO 强化学习
python train_grpo.py

# Agentic RL（多轮 Tool-Use）
python train_agent.py
```

#### Step 4：其他训练（可选）

```bash
# LoRA 微调（CPU 也能跑）
python train_lora.py

# 知识蒸馏
python train_distillation.py
```

> 所有训练脚本均支持 `--from_resume 1` 断点续训和 `--use_wandb` 训练可视化。

### 训练开销

| 模型 | 参数量 | Pretrain | SFT | RLAIF |
|------|--------|----------|-----|-------|
| TinyMind-Dense | 64M | ≈1.2h / ¥1.6 | ≈1.1h / ¥1.4 | ≈1.1h / ¥1.4 |
| TinyMind-MoE | 198M | ≈1.7h / ¥2.2 | ≈1.5h / ¥2.0 | ≈1.5h / ¥2.0 |

> 基于单卡 NVIDIA 3090 的经验估算。从零训练约 2 小时、3 块钱即可完成。

---

## 🤖 Agent 多智能体系统

项目附带一个基于 LangGraph 的 Plan-and-Execute 多智能体系统，将 TinyMind 作为 LLM 大脑，实现"先规划、再执行、可重试"的多步骤任务处理。

<!-- screenshot: agent_architecture -->
<!-- 截图文件名：images/agent_architecture.png -->
<!-- 截图说明：Agent 工作流架构图（planner → 专家 → replan → final） -->

### 架构

```
用户任务 → [planner 规划] → [专家执行循环] → [replan 复核] → [final 汇总]
                           ├── file_agent（移动文件）
                           ├── code_agent（编写代码）
                           └── story_agent（编写故事）
```

### 三种 LLM 后端

| 模式 | 说明 | 依赖 |
|------|------|------|
| `local`（默认） | 进程内加载模型权重 | torch + transformers |
| `server` | 连接 OpenAI 兼容服务 | langchain-openai |
| `mock` | 假模型，用于测试编排 | 仅 langgraph |

### 使用方式

```bash
# 命令行
python run_agent.py "写一个快速排序的Python函数"
python run_agent.py "把 inbox/note.txt 移动到 archive 目录"
python run_agent.py "写一个冒泡排序函数，然后写一个关于程序员的小故事"

# Streamlit WebUI
streamlit run agent_web.py
```

<!-- screenshot: agent_webui -->
<!-- 截图文件名：images/agent_webui.png -->
<!-- 截图说明：Agent Streamlit WebUI 界面截图，展示规划和执行过程 -->

> 详细的 Agent 使用说明请参考 [AGENT_README.md](AGENT_README.md)

---

## 🔌 模型服务与部署

### OpenAI 兼容 API

```bash
cd scripts && python serve_openai_api.py
```

支持 `reasoning_content`、`tool_calls`、`open_thinking` 等字段，可接入 FastGPT、Open-WebUI、Dify 等第三方 UI。

```bash
# 测试接口
curl http://localhost:8998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "tinymind", "messages": [{"role": "user", "content": "你好"}]}'
```

### 第三方框架

```bash
# ollama
ollama run jingyaogong/minimind-3

# vllm
vllm serve /path/to/model --served-model-name "tinymind"

# llama.cpp（需先转换格式）
python convert_hf_to_gguf.py /path/to/model
```

---

## 📊 模型参数

| 模型 | 参数量 | 词表 | 最大位置 | 层数 | 维度 | 类型 |
|------|--------|------|----------|------|------|------|
| TinyMind-Dense | 64M | 6400 | 32768 | 8 | 768 | Dense |
| TinyMind-MoE | 198M | 6400 | 32768 | 8 | 768 | MoE (4E/top-1) |

### 模型结构

<!-- screenshot: model_structure -->
<!-- 截图文件名：images/model_structure.png -->
<!-- 截图说明：TinyMind 模型结构示意图（Dense 和 MoE 两种） -->

- 采用预标准化（Pre-Norm）+ RMSNorm
- 使用 SwiGLU 激活函数
- 使用 RoPE 旋转位置编码，支持 YaRN 外推
- `q_heads=8`、`kv_heads=4`、`max_position_embeddings=32768`、`rope_theta=1e6`

---

## 📈 评估结果

> 评测框架使用 [lm-evaluation](https://github.com/EleutherAI/lm-evaluation-harness)。TinyMind 的数据规模远小于其他模型，且训练比例偏向中文，结果仅供参考。

> 具体评测数据将在后续版本中补充。

---

##  项目结构

```
TinyMind/
├── model/                          # 模型定义
│   ├── model_minimind.py           #   主体模型（Dense + MoE）
│   ├── model_lora.py               #   LoRA 低秩适配器
│   ├── tokenizer.json              #   分词器
│   └── tokenizer_config.json       #   分词器配置
├── trainer/                        # 训练脚本
│   ├── train_pretrain.py           #   预训练
│   ├── train_full_sft.py           #   监督微调
│   ├── train_lora.py               #   LoRA 微调
│   ├── train_dpo.py                #   DPO 偏好对齐
│   ├── train_ppo.py                #   PPO 强化学习
│   ├── train_grpo.py               #   GRPO / CISPO 强化学习
│   ├── train_agent.py              #   Agentic RL（多轮 Tool-Use）
│   ├── train_distillation.py       #   知识蒸馏
│   ├── train_tokenizer.py          #   分词器训练
│   ├── rollout_engine.py           #   Rollout 推理引擎
│   └── trainer_utils.py            #   训练工具集
├── dataset/                        # 数据集工具
│   └── lm_dataset.py               #   各阶段数据集类定义
├── scripts/                        # 工具脚本
│   ├── serve_openai_api.py         #   OpenAI 兼容 API 服务
│   ├── web_demo.py                 #   Streamlit 聊天 WebUI
│   ├── chat_api.py                 #   API 调用示例
│   ├── convert_model.py            #   模型格式转换
│   └── eval_toolcall.py            #   Tool Call 测试脚本
├── minimind_agent/                 # Agent 多智能体系统
│   ├── agents/                     #   三个专家 Agent
│   ├── graph/                      #   LangGraph 工作流
│   ├── config.py                   #   运行配置
│   ├── provider.py                 #   LLM 后端工厂
│   └── ...
├── tests_agent/                    # Agent 测试
├── eval_llm.py                     # 模型推理评估
├── run_agent.py                    # Agent 命令行入口
├── agent_web.py                    # Agent Streamlit WebUI
└── requirements.txt                # 依赖清单
```

---

## 📝 更新日志

<details>
<summary><b>🔥 2026-04-01</b></summary>

- 发布 TinyMind-Dense (64M) / TinyMind-MoE (198M)
- 结构对齐 Qwen3 / Qwen3-MoE 生态
- 新增 Agentic RL 训练脚本，支持多轮 Tool-Use
- 新增 Agent 多智能体系统（LangGraph 编排）
- Tokenizer 基于 BPE + ByteLevel 更新
- 新增 LoRA 权重合并导出流程

</details>

<details>
<summary><b>2025-10-24</b></summary>

- 新增 PPO、GRPO、CISPO 训练算法
- 新增断点续训功能
- 新增 YaRN 算法（RoPE 长文本外推）
- 自适应思考：`<think>` 标签动态控制推理过程

</details>

<details>
<summary><b>2025-04-26</b></summary>

- 模型参数命名对齐 Transformers 库
- 支持 llama.cpp、vllm、ollama 等第三方生态
- 统一数据集格式为 jsonl

</details>

---

## 🙏 致谢

本项目基于 [MiniMind](https://github.com/jingyaogong/minimind) 进行改造，在此感谢原作者 Jingyao Gong 的开源贡献。

实现过程中参考了以下优秀的论文与项目：

- [MobileLLM](https://arxiv.org/pdf/2402.14905) — 小模型参数分配研究
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) — MoE 架构设计
- [DeepSeekMath (GRPO)](https://arxiv.org/pdf/2402.03300) — GRPO 算法
- [CISPO](https://huggingface.co/papers/2506.13585) — Clipped Importance Sampling Policy Optimization
- [Meta LLaMA](https://github.com/meta-llama/llama3) — 模型结构参考
- [Karpathy's llama2.c](https://github.com/karpathy/llama2.c) — 极简 LLM 实现思路
- [baby-llama2-chinese](https://github.com/DLLXW/baby-llama2-chinese) — 小模型中文训练参考
- [ChatLM-mini-Chinese](https://github.com/charent/ChatLM-mini-Chinese) — 中文小模型参考
- [TinyLlama](https://github.com/jzhang38/TinyLlama) — 小模型训练参考

---

## ⚖️ 开源协议

本项目采用 [Apache License 2.0](LICENSE) 开源协议。
