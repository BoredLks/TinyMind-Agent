"""把本地 MiniMind 权重封装成一个 LangChain ChatModel（进程内推理，local 模式）。

这里复用 ``eval_llm.py`` / ``serve_openai_api.py`` 已被验证过的推理配方：
``apply_chat_template(open_thinking=False)`` → ``model.generate`` → decode → 清理思考/工具标签。
torch、transformers 以及仓库内的 ``model.model_minimind`` 都是 **惰性导入**，
因此 server / mock 模式下即使没有 torch 也能正常运行本包。
"""

from __future__ import annotations  # 启用延迟注解求值

import re                            # 正则：清理 <think> 等标签
import sys                           # 把项目根加入 sys.path 以便 import model.*
from typing import Any, Optional     # 类型标注

from langchain_core.language_models.chat_models import BaseChatModel  # ChatModel 基类
from langchain_core.messages import AIMessage, BaseMessage            # AI 消息 / 消息基类
from langchain_core.outputs import ChatGeneration, ChatResult         # _generate 的标准返回容器

from minimind_agent.config import PROJECT_ROOT  # 项目根目录（用于定位权重/分词器）


# 进程内只加载一次模型：键为权重配置，值为 (model, tokenizer, device)
_RUNTIME_CACHE: dict[tuple, tuple] = {}


def _content_to_text(content: Any) -> str:  # 把 message.content 统一转成纯文本
    """LangChain 的 message.content 可能是 str 或分块列表，统一转成纯文本。"""
    if isinstance(content, str):     # 已是字符串
        return content
    if isinstance(content, list):    # 分块列表（多模态/分段）
        parts = []
        for chunk in content:
            if isinstance(chunk, dict):  # 形如 {"type":"text","text":...}
                parts.append(str(chunk.get("text", "")))
            else:
                parts.append(str(chunk))
        return "".join(parts)        # 拼接所有片段
    return str(content)              # 其它类型兜底转字符串


def _to_minimind_messages(messages: list[BaseMessage]) -> list[dict[str, str]]:  # LangChain 消息 → MiniMind 模板输入
    """把 LangChain 消息转换成 MiniMind chat_template 接受的 [{role, content}]。"""
    role_map = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool"}  # 角色名映射
    converted: list[dict[str, str]] = []
    for message in messages:
        role = role_map.get(message.type, "user")  # 未知类型默认当作 user
        converted.append({"role": role, "content": _content_to_text(message.content)})
    return converted


def _clean_output(text: str) -> str:  # 清理生成文本里可能残留的思考标签
    """清理可能残留的思考/工具标签，返回干净正文。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)  # 删掉成对的 <think>...</think>
    text = re.sub(r"</?think>", "", text)  # 删掉落单的 <think> 或 </think>
    return text.strip()              # 去首尾空白


def _load_minimind(weight: str, hidden_size: int, num_hidden_layers: int, use_moe: bool, device: Optional[str]):  # 惰性加载模型与分词器（带缓存）
    """惰性加载 MiniMind 模型与分词器（带进程内缓存）。"""
    key = (weight, hidden_size, num_hidden_layers, use_moe, device)  # 缓存键
    if key in _RUNTIME_CACHE:        # 命中缓存直接复用（避免重复加载权重）
        return _RUNTIME_CACHE[key]

    # 仅在真正需要 local 模式时才导入这些重依赖
    import torch  # noqa: WPS433
    from transformers import AutoTokenizer  # noqa: WPS433

    # 让 "from model.model_minimind import ..." 能被找到（项目根加入 sys.path）
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: WPS433

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")  # 设备：显式优先，否则自动
    tokenizer = AutoTokenizer.from_pretrained(str(PROJECT_ROOT / "model"))  # 从 model/ 加载分词器（含 chat_template）
    model = MiniMindForCausalLM(     # 用配置实例化模型
        MiniMindConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            use_moe=bool(use_moe),
        )
    )
    moe_suffix = "_moe" if use_moe else ""  # MoE 权重文件名带 _moe 后缀
    ckpt = PROJECT_ROOT / "out" / f"{weight}_{hidden_size}{moe_suffix}.pth"  # 权重路径，如 out/full_sft_768.pth
    if not ckpt.exists():            # 权重缺失：给出清晰提示（并建议改用 server/mock）
        raise FileNotFoundError(
            f"找不到 MiniMind 权重: {ckpt}\n请确认 out/ 目录下存在 {ckpt.name}（或改用 server / mock 模式）。"
        )
    state_dict = torch.load(str(ckpt), map_location=resolved_device)  # 读取权重到目标设备
    model.load_state_dict(state_dict, strict=True)  # 严格加载（键必须完全匹配）
    # GPU 上用半精度更快；CPU 上保持 float32 以保证算子兼容与稳定
    model = model.half() if resolved_device == "cuda" else model.float()
    model = model.eval().to(resolved_device)  # 评估模式 + 搬到设备

    _RUNTIME_CACHE[key] = (model, tokenizer, resolved_device)  # 写入缓存
    return _RUNTIME_CACHE[key]


class MiniMindChatModel(BaseChatModel):  # LangChain ChatModel：进程内调用本地 MiniMind 权重
    """LangChain ChatModel：进程内调用本地 MiniMind 权重生成文本。"""

    weight: str = "full_sft"         # 权重前缀（pydantic 字段，可由 create_model 传入）
    hidden_size: int = 768           # 隐藏维度
    num_hidden_layers: int = 8       # 层数
    use_moe: bool = False            # 是否 MoE
    device: Optional[str] = None     # 设备（None=自动）
    temperature: float = 0.6         # 采样温度
    top_p: float = 0.9               # 核采样阈值
    max_new_tokens: int = 1024       # 最大生成长度

    @property
    def _llm_type(self) -> str:      # ChatModel 要求实现：返回模型类型标识
        return "minimind-local"

    def _generate(                   # ChatModel 的核心方法：消息 → 一次生成
        self,
        messages: list[BaseMessage],  # 输入消息
        stop: Optional[list[str]] = None,  # 停止词（此实现未用）
        run_manager: Any = None,     # 回调管理器（此实现未用）
        **kwargs: Any,
    ) -> ChatResult:
        import torch  # noqa: WPS433  # 惰性导入 torch（local 模式才需要）

        model, tokenizer, device = _load_minimind(  # 取（缓存的）模型/分词器/设备
            self.weight, self.hidden_size, self.num_hidden_layers, self.use_moe, self.device
        )
        conversation = _to_minimind_messages(messages)  # 转成模板输入
        prompt = tokenizer.apply_chat_template(  # 套 chat 模板，得到喂给模型的 prompt 文本
            conversation,
            tokenize=False,
            add_generation_prompt=True,  # 末尾补“助手开始”提示
            open_thinking=False,  # 小模型上“思考 + 任务”不稳定，关闭自适应思考
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)  # 编码并搬到设备

        # temperature<=0 时退化为贪心，避免自定义 generate 里的除零
        do_sample = self.temperature > 0  # 温度>0 才采样
        temperature = self.temperature if self.temperature > 0 else 1.0  # 贪心时温度用 1.0 占位（不参与）

        with torch.no_grad():        # 推理不需要梯度
            generated = model.generate(  # 调用 MiniMind 自定义的自回归生成
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=self.top_p,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0][len(inputs["input_ids"][0]):]  # 只取新生成部分（去掉 prompt）
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)  # 解码成文本
        text = _clean_output(raw)    # 清理思考标签

        message = AIMessage(content=text)  # 包成 AI 消息
        return ChatResult(generations=[ChatGeneration(message=message)])  # 返回标准结果
