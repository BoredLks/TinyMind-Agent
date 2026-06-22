from torch.utils.data import Dataset  # PyTorch 数据集基类（实现 __len__ / __getitem__ 即可被 DataLoader 使用）
import torch                          # 张量
import json                           # 解析对话中以字符串形式存放的 tools / tool_calls
import os                             # 设置环境变量
import random                         # 概率性加 system、概率性去思考标签
from datasets import load_dataset, Features, Sequence, Value  # HF datasets：高效读取 jsonl 与声明字段类型
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 关闭分词器多进程并行，避免与 DataLoader 多进程冲突时报警

def pre_processing_chat(conversations, add_system_ratio=0.2):  # 对话预处理：按概率给开头加一条 system 提示
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations  # 含工具的样本原样返回，不加 system

    SYSTEM_PROMPTS = [                # 可随机选用的 system 提示（中英混合，增强鲁棒性）
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':  # 开头还没有 system
        if random.random() < add_system_ratio:    # 以 add_system_ratio 概率添加
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations  # 头部插入随机 system
    return conversations              # 否则原样返回

def post_processing_chat(prompt_content, empty_think_ratio=0.2):  # 对拼好的 prompt 文本做后处理：概率性删除“空思考”块
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:  # 命中空思考块且落在 80% 区间
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')  # 删掉空思考块（让模型学会“可不思考”）
    return prompt_content

class PretrainDataset(Dataset):       # 预训练数据集：纯文本，做 next-token 续写
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length  # 序列定长
        self.samples = load_dataset('json', data_files=data_path, split='train')  # 读取 jsonl（每条含 'text'）

    def __len__(self):
        return len(self.samples)      # 样本数

    def __getitem__(self, index):
        sample = self.samples[index]  # 取一条
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids  # 编码（留出 bos/eos 两个位置）
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]  # 首尾加 bos/eos
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))  # 右侧 pad 到定长
        input_ids = torch.tensor(input_ids, dtype=torch.long)  # 转张量
        labels = input_ids.clone()    # 标签与输入相同（预训练对全序列算损失）
        labels[input_ids == self.tokenizer.pad_token_id] = -100  # pad 位置置 -100（交叉熵忽略）
        return input_ids, labels      # 返回 (输入, 标签)；训练时模型内部会自动右移一位对齐


class SFTDataset(Dataset):            # 监督微调数据集：多轮对话，仅对“助手回答”部分计损失
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})  # 显式声明字段类型，保证不同样本字段对齐
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)  # 读取
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids  # “助手回答开始”的标记序列（<|im_start|>assistant\n）
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids  # “回答结束”的标记序列（<|im_end|>\n）

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):  # 把一段对话套上 chat 模板，得到训练用的纯文本
        messages = []
        tools = None
        for message in conversations:  # 逐条处理，顺便把字符串化的 tools/tool_calls 解析回对象
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):  # system 里带 tools → 提取出来传给模板
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):  # 助手的 tool_calls 若是字符串则解析
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(  # 套模板（不加生成提示，因为这是带答案的训练样本）
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):  # 生成标签：只对“助手回答段”保留真实 token，其余置 -100（不计损失）
        labels = [-100] * len(input_ids)   # 默认全部忽略
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:  # 找到“助手开始”标记
                start = i + len(self.bos_id)  # 回答内容起点
                end = start
                while end < len(input_ids):   # 向后找“回答结束”标记
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):  # 回答段（含结束标记）设为可学习
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)  # 跳到本段之后继续找下一段助手回答
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])  # 概率性加 system
        prompt = self.create_chat_prompt(conversations)  # 套模板
        prompt = post_processing_chat(prompt)  # 概率性去空思考块
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]  # 编码并截断
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))  # 右侧 pad 到定长
        labels = self.generate_labels(input_ids)  # 仅助手回答段计损失
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)  # 返回 (输入, 标签)


class DPODataset(Dataset):            # DPO 偏好对齐数据集：每条含 chosen(更优) 与 rejected(更差) 两段对话
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0  # pad id（无则用 0）
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids  # 助手开始标记
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids  # 回答结束标记
        self.samples = load_dataset('json', data_files=file_path, split='train')  # 读取

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = sample['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(  # chosen 套模板
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)  # 概率性去空思考块

        rejected_prompt = self.tokenizer.apply_chat_template(  # rejected 套模板
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)
        chosen_encoding = self.tokenizer(  # 编码到定长（截断 + pad）
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)  # 标记 chosen 的助手回答段（用于只在回答处算 DPO 损失）

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)  # 输入（去掉最后一个 token）
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)   # 目标（右移一位）
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)  # 与目标对齐的损失掩码
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {                       # 一条 DPO 样本同时含 chosen 与 rejected 的输入/目标/掩码
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):  # 与 SFT 的 generate_labels 同理，但返回 0/1 掩码（1=助手回答段）
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:  # 助手开始
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):   # 找回答结束
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):  # 回答段置 1
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):          # 在线 RL（PPO/GRPO）数据集：只取“到用户为止”的 prompt，答案在训练时在线采样
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio  # 按概率开启 thinking
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids  # 助手开始（此处不含 \n）
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids  # 结束标记

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):  # 构造采样用 prompt：丢掉最后一条（标准答案），加生成提示让模型续写
        conversations = pre_processing_chat(conversations)  # 概率性加 system
        use_thinking = random.random() < self.thinking_ratio  # 概率性开启思考
        return self.tokenizer.apply_chat_template(
            conversations[:-1],        # 去掉最后一条（即去掉参考答案）
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True  # 末尾加“助手开始”提示，供模型续写
        )
    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])

        return {                       # 只返回 prompt（答案在线生成后由奖励函数评分）
            'prompt': prompt,
            'answer': ""
        }

class AgentRLDataset(Dataset):        # Agentic RL 数据集：带工具定义与标准答案 gt，用于训练模型“正确调用工具”
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:  # 逐行读取 jsonl
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):  # 拆出“到用户为止”的消息与工具定义
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):  # 取出 system 中的工具
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        return messages[:-1], tools    # 去掉最后一条（标准答案），返回消息与工具

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}  # gt 为标准答案，供奖励判定


if __name__ == "__main__":            # 作为脚本直接运行时无操作（本文件主要被训练脚本导入使用）
    pass
