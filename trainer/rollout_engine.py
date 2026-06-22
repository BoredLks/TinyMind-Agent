"""
# 如果使用sglang加速，需通过以下命令首先启动（transformers格式）模型：
python -m sglang.launch_server --model-path ./minimind-3 --attention-backend triton --host 0.0.0.0 --port 8998
"""
import os                            # 路径处理
import sys                           # 修改模块搜索路径

__package__ = "trainer"             # 显式声明所属包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 项目根加入搜索路径

import requests                      # 调用 SGLang 的 HTTP 推理接口
import torch                         # 张量与生成
import torch.distributed as dist     # 分布式（同步权重/广播）
from abc import ABC, abstractmethod  # 抽象基类，定义引擎统一接口
from contextlib import nullcontext   # 无操作上下文（无 autocast 时占位）
from dataclasses import dataclass    # 用 dataclass 封装 rollout 结果
from typing import List, Optional, Tuple  # 类型注解
from torch import Tensor             # 张量类型注解
from torch.nn.parallel import DistributedDataParallel  # DDP 解包
from transformers import AutoTokenizer  # 分词器


# ===== 计算每个 token 的 logprob =====
def compute_per_token_logps(model, input_ids: Tensor, n_keep: int, attention_mask: Optional[Tensor] = None) -> Tensor:  # 计算序列末尾 n_keep 个 token 的对数概率（PPO/GRPO 需要）
    if n_keep <= 0:                  # 不需要保留任何 token
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)  # 返回空张量
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model  # 解开 DDP
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids  # 若处于推理张量则克隆，避免梯度报错
    logits = unwrapped(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]  # 只算末尾 n_keep+1 个位置的 logits，并丢最后一个（预测错位对齐）
    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):  # 逐样本：取末尾 n_keep 个真实 token 的 logprob
        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)  # log_softmax 后按真实 token 索引 gather 出对数概率
        )
    return torch.stack(per_token_logps)  # 堆叠成 [B, n_keep]


# ===== Rollout 结果 =====
@dataclass
class RolloutResult:                 # 一次采样（rollout）产出的结构化结果
    output_ids: Tensor               # 完整序列（prompt + 续写）
    completion_ids: Tensor           # 仅续写部分
    per_token_logps: Tensor          # 续写每个 token 的对数概率
    completions: List[str]           # 解码后的续写文本
    prompt_lens: Tensor              # 各样本的 prompt 长度
    completion_mask: Tensor          # 续写有效位置掩码（1=有效, 0=pad）


# ===== Rollout 引擎抽象基类 =====
class RolloutEngine(ABC):            # 统一接口：不同后端（torch / sglang）都实现 rollout 与 update_policy
    tokenizer = None

    @abstractmethod
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:  # 给每个 prompt 采样 num_generations 条续写
        pass

    @abstractmethod
    def update_policy(self, model: torch.nn.Module):  # 把最新策略模型权重同步到推理引擎
        pass


# ===== PyTorch 原生推理引擎 =====
class TorchRolloutEngine(RolloutEngine):  # 直接用 PyTorch 的 model.generate 做采样（简单，但比专用推理引擎慢）
    def __init__(self, policy_model: torch.nn.Module, tokenizer, device: str = "cuda", autocast_ctx=None):
        self.policy_model = policy_model  # 策略模型（被训练的模型）
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_ctx = autocast_ctx  # 混合精度上下文（可选）

    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        model = self.policy_model.module if isinstance(self.policy_model, DistributedDataParallel) else self.policy_model  # 解 DDP
        ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()  # 有 autocast 用之，否则空上下文
        with torch.no_grad(), ctx:   # 采样不需要梯度
            output_ids = model.generate(  # 每个 prompt 复制 num_generations 份后一起生成
                input_ids=prompt_ids.repeat_interleave(num_generations, dim=0),
                attention_mask=attention_mask.repeat_interleave(num_generations, dim=0),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            ).clone()  # [B*num_gen, P+R]
            prompt_len = prompt_ids.size(1)  # prompt 长度（定长）
            completion_ids = output_ids[:, prompt_len:]  # [B*num_gen, R]
            full_mask = (output_ids != self.tokenizer.pad_token_id).long()  # 整条序列的有效掩码
            per_token_logps = compute_per_token_logps(self.policy_model, output_ids, completion_ids.size(1), attention_mask=full_mask)  # 算续写每 token 的 logprob
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)  # 解码成文本
        return RolloutResult(output_ids, completion_ids, per_token_logps, completions,
                             prompt_ids.new_full((output_ids.size(0),), prompt_len),  # 所有样本 prompt 长度相同
                             attention_mask.new_ones(output_ids.size(0), completion_ids.size(1)))  # torch 后端续写全部视为有效

    def update_policy(self, model: torch.nn.Module):  # torch 后端直接换引用即可（推理与训练共用同一模型）
        self.policy_model = model


# ===== SGLang HTTP API 推理引擎 =====
class SGLangRolloutEngine(RolloutEngine):  # 通过 HTTP 调用独立的 SGLang 服务做高效采样（适合大批量 RL）
    def __init__(self, base_url: str, model_path: str, shared_ckpt_path: str = "./sglang_ckpt", timeout: int = 120):
        self.base_url = base_url.rstrip('/')  # 服务地址（去尾部斜杠）
        self.shared_ckpt_path = shared_ckpt_path  # 同步权重时落盘的共享路径
        self.timeout = timeout       # HTTP 超时
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  # 分词器
        self.http = requests         # HTTP 客户端

    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        # 去除左侧 padding tokens，只保留有效 token
        input_ids_list = []
        for ids, mask in zip(prompt_ids, attention_mask):  # 用 mask 还原出每个 prompt 的真实 token（去掉左 pad）
            valid_ids = ids[mask.bool()].tolist()
            input_ids_list.append(valid_ids)
        all_input_ids = [ids for ids in input_ids_list for _ in range(num_generations)]  # 每个 prompt 复制 num_generations 份

        payload = {                  # SGLang /generate 接口的请求体
            "input_ids": all_input_ids,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else [],  # 遇 eos 停止
            },
            "return_logprob": True,  # 让服务返回每 token 的 logprob
        }

        resp = self.http.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)  # 发起生成请求
        resp.raise_for_status()      # 非 2xx 抛异常

        results = resp.json()        # 解析返回
        if not isinstance(results, list):  # 单条结果也统一成列表
            results = [results]

        all_output_ids, all_completion_ids, all_logprobs = [], [], []  # 收集各样本结果
        completions = []

        for i, result in enumerate(results):  # 逐条解析
            meta = result.get("meta_info", {})
            completion_ids = meta.get("output_ids", result.get("output_ids", []))  # 续写 token ids
            raw_logprobs = meta.get("output_token_logprobs", [])  # 续写 logprobs（格式可能是 [(lp, tok_id, ...), ...]）

            logprobs = []
            for item in raw_logprobs:  # 兼容多种返回格式，取出纯 logprob 数值
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    logprobs.append(item[0])
                elif isinstance(item, (int, float)):
                    logprobs.append(item)

            if len(logprobs) < len(completion_ids):  # logprob 不够则在前面补 0 对齐
                logprobs = [0.0] * (len(completion_ids) - len(logprobs)) + logprobs
            elif len(logprobs) > len(completion_ids):  # 过多则取末尾对齐
                logprobs = logprobs[-len(completion_ids):] if completion_ids else []
            prompt = all_input_ids[i]
            full_output = prompt + completion_ids  # 拼成完整序列
            all_output_ids.append(full_output)
            all_completion_ids.append(completion_ids)
            all_logprobs.append(logprobs)
            completions.append(self.tokenizer.decode(completion_ids, skip_special_tokens=True))  # 解码文本

        device = prompt_ids.device   # 目标设备
        max_comp_len = max(1, max(len(ids) for ids in all_completion_ids))  # 续写的最大长度（至少 1）
        max_out_len = max(len(ids) for ids in all_input_ids) + max_comp_len  # 完整序列最大长度

        def pad_to_tensor(seqs, max_len, pad_val=0):  # 把变长序列右侧 pad 成等长张量
            return torch.tensor([s + [pad_val] * (max_len - len(s)) for s in seqs], device=device)

        pad_id = self.tokenizer.pad_token_id  # pad id
        return RolloutResult(        # 组装与 torch 后端一致的结果结构
            output_ids=pad_to_tensor(all_output_ids, max_out_len, pad_val=pad_id),
            completion_ids=pad_to_tensor(all_completion_ids, max_comp_len, pad_val=pad_id),
            per_token_logps=pad_to_tensor(all_logprobs, max_comp_len, pad_val=0.0),
            completions=completions,
            prompt_lens=torch.tensor([len(ids) for ids in all_input_ids], device=device),
            completion_mask=torch.tensor([[1] * len(ids) + [0] * (max_comp_len - len(ids)) for ids in all_completion_ids], device=device),  # 真实续写位置为 1，pad 为 0
        )

    def update_policy(self, model: torch.nn.Module):  # 把训练中的最新权重落盘并通知 SGLang 服务热加载
        ok = True
        if not dist.is_initialized() or dist.get_rank() == 0:  # 仅主进程负责落盘与通知
            try:
                unwrapped = model.module if isinstance(model, DistributedDataParallel) else model  # 解 DDP
                unwrapped = getattr(unwrapped, '_orig_mod', unwrapped)  # 解 compile
                abs_path = os.path.abspath(self.shared_ckpt_path)  # 共享绝对路径
                state_dict = {k: v.detach().half().cpu() for k, v in unwrapped.state_dict().items()}  # 取 fp16/CPU 权重
                unwrapped.save_pretrained(abs_path, state_dict=state_dict, safe_serialization=False)  # 保存为 transformers 格式
                self.tokenizer.save_pretrained(abs_path)  # 一并保存分词器
                resp = self.http.post(f"{self.base_url}/update_weights_from_disk", json={"model_path": abs_path}, timeout=self.timeout)  # 通知服务从磁盘热更新权重
                if resp.status_code != 200: print(f"[SGLANG WARNING] update_weights 失败: {resp.status_code}, {resp.text}")  # 失败告警
                ok = resp.status_code == 200
            except Exception as e:
                print(f"[SGLANG WARNING] update_weights 异常: {e}"); ok = False  # 异常也标记失败
        if dist.is_initialized():    # 分布式下把主进程的成功标志广播给所有进程并同步
            ok_t = torch.tensor(int(ok), device=next(model.parameters()).device)
            dist.broadcast(ok_t, src=0); dist.barrier(); ok = bool(ok_t.item())
        if not ok: raise RuntimeError("SGLang update_policy failed")  # 任一进程失败则报错
        return ok

    def flush_cache(self) -> bool:   # 清空 SGLang 的 KV 缓存（换权重后需要）
        resp = self.http.post(f"{self.base_url}/flush_cache", timeout=30)
        return resp.status_code == 200

    def health(self) -> bool:        # 健康检查：服务是否可用
        try:
            resp = self.http.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except:
            return False


# ===== 工厂函数 =====
def create_rollout_engine(           # 按类型创建对应的 rollout 引擎
    engine_type: str = "torch",
    policy_model: torch.nn.Module = None,
    tokenizer = None,
    device: str = "cuda",
    autocast_ctx = None,
    sglang_base_url: str = None,
    sglang_model_path: str = None,
    sglang_shared_path: str = None,
) -> RolloutEngine:
    if engine_type == "torch":       # PyTorch 原生
        return TorchRolloutEngine(policy_model, tokenizer, device, autocast_ctx)
    elif engine_type == "sglang":    # SGLang HTTP
        return SGLangRolloutEngine(sglang_base_url, sglang_model_path, sglang_shared_path)
    else:
        raise ValueError(f"不支持的引擎类型: {engine_type}")  # 未知类型
