# 本文件使 model/ 成为一个 Python 包（package），从而支持
# `from model.model_minimind import MiniMindConfig, MiniMindForCausalLM` 这样的导入写法。
# MiniMind 把“模型结构定义”集中放在该包内：
#   - model_minimind.py：MiniMind 主体（配置 / 旋转位置编码 / 注意力 / 前馈 / MoE / 因果语言模型 + 生成）
#   - model_lora.py    ：LoRA 低秩适配器的实现，以及挂载/加载/保存/合并的工具函数
# 该包无需在导入时执行任何初始化代码，因此保持空实现即可（仅靠本文件存在来标识这是一个包）。
