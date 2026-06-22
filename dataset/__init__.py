# 本文件使 dataset/ 成为一个 Python 包，从而支持 `from dataset.lm_dataset import ...`。
# 该包内的 lm_dataset.py 定义了各训练阶段所需的数据集类：
#   - PretrainDataset ：预训练（纯文本续写）
#   - SFTDataset      ：监督微调（多轮对话/工具调用，仅对“助手回答”计损失）
#   - DPODataset      ：偏好对齐（chosen/rejected 成对样本）
#   - RLAIFDataset    ：在线 RL（PPO/GRPO）只取 prompt，答案在线采样
#   - AgentRLDataset  ：Agentic RL（带工具与标准答案 gt）
# 这里无需初始化代码，保持空实现即可（仅靠文件存在标识这是一个包）。
