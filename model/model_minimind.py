import math, torch, torch.nn.functional as F  # math：数学函数；torch：张量与自动求导；F：函数式算子（softmax/激活/注意力等）
from torch import nn                              # nn：神经网络层与模块基类
from transformers.activations import ACT2FN      # ACT2FN：按名字("silu"等)取激活函数的字典
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig  # HF 基类：模型基类 / 生成混入 / 配置基类
from transformers.modeling_outputs import MoeCausalLMOutputWithPast          # HF 标准输出容器（含 loss/logits/past 等字段）

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):           # 模型超参配置类，继承 HF 的 PretrainedConfig（可被 save_pretrained/from_pretrained 序列化）
    model_type = "minimind"                       # 模型类型标识，HF 用它把配置与模型类关联起来
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):  # 三个最常改的超参显式列出，其余走 kwargs
        super().__init__(**kwargs)                # 先让基类处理通用字段（如 pad/bos/eos、torch_dtype 等）
        self.hidden_size = hidden_size            # 隐藏层维度（每个 token 向量的长度）
        self.num_hidden_layers = num_hidden_layers  # Transformer 层数（堆叠的 MiniMindBlock 个数）
        self.use_moe = use_moe                    # 是否使用 MoE（混合专家）前馈；False 则用普通 FFN
        self.dropout = kwargs.get("dropout", 0.0)  # dropout 比例（默认 0，推理/小模型通常关闭）
        self.vocab_size = kwargs.get("vocab_size", 6400)  # 词表大小（与分词器一致，MiniMind 为 6400）
        self.bos_token_id = kwargs.get("bos_token_id", 1)  # 句首特殊 token id（<|im_start|>）
        self.eos_token_id = kwargs.get("eos_token_id", 2)  # 句末特殊 token id（<|im_end|>），生成时遇到它停止
        self.flash_attn = kwargs.get("flash_attn", True)  # 是否启用 Flash Attention（scaled_dot_product_attention 加速）
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)  # 注意力头数（Query 头数）
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)  # KV 头数（<Q 头数即为 GQA 分组查询注意力，省显存）
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)  # 每个头的维度，默认 hidden/头数
        self.hidden_act = kwargs.get("hidden_act", 'silu')  # FFN 激活函数名（SiLU，配合 SwiGLU 门控）
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)  # FFN 中间维度，按 ~π 倍并对齐到 64 的整数倍
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)  # 预计算 RoPE 的最大位置长度
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)  # RMSNorm 防止除零的 epsilon
        self.rope_theta = kwargs.get("rope_theta", 1e6)  # RoPE 频率基数 θ（越大，长程位置区分越平滑）
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)  # 是否共享输入词嵌入与输出 lm_head 权重（省参数）
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)  # 推理时是否启用 RoPE 外推（YaRN），扩展上下文长度
        self.rope_scaling = {                     # YaRN 外推的具体参数（仅当 inference_rope_scaling=True 时生效）
            "beta_fast": 32,                      # 高频维度的边界（基本不缩放，保留局部分辨率）
            "beta_slow": 1,                       # 低频维度的边界（缩放最多，扩展长程）
            "factor": 16,                         # 外推倍数（把有效长度放大约 16 倍）
            "original_max_position_embeddings": 2048,  # 训练时的原始上下文长度
            "attention_factor": 1.0,              # 注意力缩放因子（YaRN 对 logits 的温度修正）
            "type": "yarn"                        # 外推方法类型
        } if self.inference_rope_scaling else None  # 不启用外推则为 None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)  # 专家总数
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)  # 每个 token 实际激活的专家数（top-k 的 k）
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)  # 单个专家 FFN 的中间维度
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)  # 是否把 top-k 专家权重重新归一化到和为 1
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)  # 路由负载均衡辅助损失的权重系数

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):                   # RMSNorm：只用“均方根”做归一化（不减均值），比 LayerNorm 更省算力、效果相当
    def __init__(self, dim: int, eps: float = 1e-5):  # dim：归一化的特征维度；eps：数值稳定项
        super().__init__()
        self.eps = eps                            # 保存 epsilon
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的逐通道缩放参数 γ，初始化为全 1

    def norm(self, x):                            # 归一化主体：x / sqrt(mean(x^2)+eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # rsqrt=1/sqrt；按最后一维求均方再开方取倒数

    def forward(self, x):                         # 前向
        return (self.weight * self.norm(x.float())).type_as(x)  # 先升 float32 归一化（更稳），乘缩放后再转回原 dtype

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):  # 预计算 RoPE 旋转位置编码所需的 cos/sin 表
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0  # 每对维度的基础角频率 1/θ^(2i/d)；attn_factor 默认 1
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp  # 启用 YaRN 外推时，按频段对频率做插值缩放
        orig_max, factor, beta_fast, beta_slow, attn_factor = (  # 取出 YaRN 的各项参数
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),  # 原始长度、外推倍数
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)  # 高/低频边界、注意力缩放
        )
        if end / orig_max > 1.0:                  # 只有当目标长度确实超过原始长度时才需要外推
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))  # 由“波长边界 b”反解对应的维度下标
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)  # 计算需要插值的维度区间 [low, high]
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)  # 在该区间内线性从 0 渐变到 1（γ）
            freqs = freqs * (1 - ramp + ramp / factor)  # 高频维度几乎不变、低频维度被缩放 1/factor，实现平滑外推
    t = torch.arange(end, device=freqs.device)    # 位置索引 0..end-1
    freqs = torch.outer(t, freqs).float()         # 外积：得到 [位置, 半维] 的角度矩阵（每个位置×每个频率）
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor  # cos 表，拼成两半以匹配 rotate_half 的成对旋转
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor  # sin 表，同上
    return freqs_cos, freqs_sin                   # 返回供注意力使用的 cos/sin 位置编码表

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):  # 把 RoPE 旋转作用到 query/key 上（位置信息以“旋转”形式注入）
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)  # 把后半维取负移到前面：实现复数 i·x 的等价旋转
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)  # q' = q·cos + rotate_half(q)·sin
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)  # k' = k·cos + rotate_half(k)·sin
    return q_embed, k_embed                       # 返回带位置信息的 q、k

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:  # GQA 用：把较少的 KV 头复制 n_rep 份，对齐到 Q 头数
    bs, slen, num_key_value_heads, head_dim = x.shape  # 形状：[批, 序列长, KV头数, 头维]
    if n_rep == 1: return x                       # 不需要复制（即标准多头注意力 MHA）则原样返回
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))  # 在头维后插一维并扩展，再展平成 num_kv*n_rep 个头

class Attention(nn.Module):                       # 自注意力模块（GQA + QK-Norm + RoPE + 可选 Flash/KV缓存）
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads  # KV 头数（None 则退化为 MHA）
        self.n_local_heads = config.num_attention_heads  # Q 头数
        self.n_local_kv_heads = self.num_key_value_heads  # KV 头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 每个 KV 头要被多少个 Q 头共享（复制倍数）
        self.head_dim = config.head_dim           # 单头维度
        self.is_causal = True                     # 因果注意力：每个位置只能看到自己及左侧（自回归语言模型）
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)  # 生成 Q 的线性投影
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)    # 生成 K（头数较少）
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)    # 生成 V（头数较少）
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)  # 注意力输出再投影回 hidden 维
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # 对 Q 每个头做 RMSNorm（QK-Norm，稳定训练）
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # 对 K 每个头做 RMSNorm
        self.attn_dropout = nn.Dropout(config.dropout)  # 注意力权重上的 dropout（手动路径用）
        self.resid_dropout = nn.Dropout(config.dropout)  # 输出残差上的 dropout
        self.dropout = config.dropout             # dropout 比例（Flash 路径用）
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn  # 当前 PyTorch 支持 SDPA 且配置开启则用 Flash

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):  # x:[B,L,H]
        bsz, seq_len, _ = x.shape                 # 批大小、当前序列长度
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)  # 分别投影出 Q/K/V
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)     # 拆成多头：[B,L,Q头,头维]
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  # [B,L,KV头,头维]
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  # [B,L,KV头,头维]
        xq, xk = self.q_norm(xq), self.k_norm(xk)  # 对 Q、K 施加 QK-Norm
        cos, sin = position_embeddings            # 取出当前位置对应的 RoPE cos/sin
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)  # 给 Q、K 注入旋转位置编码
        if past_key_value is not None:            # 增量解码：把历史的 K、V 缓存拼到当前之前
            xk = torch.cat([past_key_value[0], xk], dim=1)  # 拼历史 K
            xv = torch.cat([past_key_value[1], xv], dim=1)  # 拼历史 V
        past_kv = (xk, xv) if use_cache else None  # 需要缓存则把（含历史的）K、V 返回，供下一步复用
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))  # KV 复制到 Q 头数，并转成 [B,头,L,头维]
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):  # 满足条件走 Flash 高效注意力
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)  # 由底层算子完成缩放点积+因果掩码+softmax
        else:                                     # 否则走手写注意力（兼容缓存解码、自定义 mask 等情形）
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)  # 注意力打分 QK^T/√d
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)  # 加上严格上三角 -inf，屏蔽未来位置
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9  # padding 位置打成 -1e9（softmax 后≈0）
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv  # softmax 得权重（float32 更稳）→ dropout → 加权求和 V
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)  # 头维合并回去：[B,L,Q头*头维]
        output = self.resid_dropout(self.o_proj(output))  # 输出投影回 hidden 维并做残差 dropout
        return output, past_kv                    # 返回注意力输出与（可选的）KV 缓存

class FeedForward(nn.Module):                     # 普通前馈网络（SwiGLU 门控结构）
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size  # 中间维度，默认取配置值
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)  # 门控分支（过激活）
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)  # 降维回 hidden
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)    # 值分支（不过激活）
        self.act_fn = ACT2FN[config.hidden_act]   # 激活函数（SiLU）

    def forward(self, x):                         # SwiGLU：down( act(gate(x)) * up(x) )
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))  # 门控分支激活后逐元素乘值分支，再降维

class MOEFeedForward(nn.Module):                  # 混合专家前馈（MoE）：每个 token 只激活 top-k 个专家
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)  # 路由器：给每个专家打分
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])  # 一组并列的专家 FFN
        self.act_fn = ACT2FN[config.hidden_act]   # 激活函数（与专家内部一致）

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape  # 记录原始形状以便最后还原
        x_flat = x.view(-1, hidden_dim)            # 把 [B,L,H] 摊平成 [B*L, H]，逐 token 处理
        scores = F.softmax(self.gate(x_flat), dim=-1)  # 路由打分→softmax，得到每个 token 对各专家的概率
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)  # 取每个 token 概率最高的 k 个专家及其权重
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)  # 把这 k 个权重重新归一化到和为 1
        y = torch.zeros_like(x_flat)               # 输出累加器，与 x_flat 同形
        for i, expert in enumerate(self.experts):  # 逐个专家处理被路由到它的 token
            mask = (topk_idx == i)                 # 标记哪些 (token, 槽位) 选中了第 i 个专家
            if mask.any():                         # 有 token 选了该专家
                token_idx = mask.any(dim=-1).nonzero().flatten()  # 这些 token 的行索引
                weight = topk_weight[mask].view(-1, 1)  # 对应的路由权重
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))  # 该专家输出乘权重，按索引累加回 y
            elif self.training:                    # 训练时若该专家没被任何 token 选中
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())  # 加“乘 0 的占位项”，让该专家参数仍进入计算图（避免 DDP 报未用参数）
        if self.training and self.config.router_aux_loss_coef > 0:  # 训练时计算负载均衡辅助损失，鼓励专家被均匀使用
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)  # 各专家被选中的平均频率
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef  # 负载×平均概率之和×系数
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()  # 推理/不启用时辅助损失为 0
        return y.view(batch_size, seq_len, hidden_dim)  # 还原成 [B,L,H]

class MiniMindBlock(nn.Module):                   # 一层 Transformer 块（Pre-Norm：先归一化再进子层，子层输出做残差相加）
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)        # 自注意力子层
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)          # 注意力前的归一化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 前馈前的归一化
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)      # 前馈：普通 FFN 或 MoE

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states                  # 保存输入用于残差连接
        hidden_states, present_key_value = self.self_attn(  # 归一化后进注意力
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual                 # 残差：注意力输出 + 原输入
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))  # 再过（归一化后的）前馈并残差
        return hidden_states, present_key_value   # 返回本层输出与 KV 缓存

class MiniMindModel(nn.Module):                   # MiniMind 主干（embedding + N 层 Block + 末端归一化），不含输出头
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers  # 词表大小与层数
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)  # 词嵌入：token id → 向量
        self.dropout = nn.Dropout(config.dropout)  # 嵌入后的 dropout
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])  # 堆叠 N 层 Block
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最后一层输出的归一化
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)  # 预计算整段 RoPE 表
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)  # 注册为 buffer（随模型搬设备但不保存进权重文件）
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape  # [B, L]
        if hasattr(past_key_values, 'layers'): past_key_values = None  # 兼容 transformers 新版传入的 Cache 对象：这里不支持，置空
        past_key_values = past_key_values or [None] * len(self.layers)  # 没有缓存则给每层一个 None 占位
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0  # 增量解码时，当前 token 的起始位置 = 已缓存长度
        hidden_states = self.dropout(self.embed_tokens(input_ids))  # 词嵌入 + dropout
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:             # 若 buffer 在 meta 设备初始化时被清零，则现场重算一遍
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)  # 重算后搬到当前设备
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])  # 取本次 token 对应的位置编码切片
        presents = []                             # 收集各层 KV 缓存
        for layer, past_key_value in zip(self.layers, past_key_values):  # 逐层前向
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)              # 记录该层 KV
        hidden_states = self.norm(hidden_states)  # 末端归一化
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())  # 汇总所有 MoE 层的辅助损失（无 MoE 则为 0）
        return hidden_states, presents, aux_loss  # 返回隐藏态、各层 KV、辅助损失

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):  # 因果语言模型：主干 + 输出头，并提供生成能力
    config_class = MiniMindConfig                 # 关联配置类（HF 据此创建/加载）
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}  # 声明 lm_head 与词嵌入共享权重的键映射
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()  # 未传配置则用默认配置
        super().__init__(self.config)             # 初始化 HF 基类
        self.model = MiniMindModel(self.config)   # 主干网络
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)  # 输出头：hidden → 词表 logits
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight  # 绑定权重共享（输入嵌入=输出头）
        self.post_init()                          # HF 的统一权重初始化/收尾

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)  # 走主干得到隐藏态
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep  # 只对末尾若干位置算 logits（生成时只需最后一个，省算力）
        logits = self.lm_head(hidden_states[:, slice_indices, :])  # 隐藏态 → 词表 logits
        loss = None                               # 默认无损失（推理）
        if labels is not None:                    # 训练：提供了标签则算交叉熵
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()  # 预测第 t 个用前 t-1，标签整体右移一位（next-token 预测）
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)  # 展平后算交叉熵，-100 的位置不计损失（如 padding/prompt）
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)  # 打包成 HF 标准输出

    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()                       # 整个生成过程关闭梯度，省显存提速
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):  # 自定义自回归采样（覆盖 HF 默认 generate）
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)  # 起始 token 序列；按返回序列数复制成多份
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None  # mask 同步复制
        past_key_values = kwargs.pop("past_key_values", None)  # 初始无 KV 缓存
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)  # 每条序列是否已生成结束的标记
        if streamer: streamer.put(input_ids.cpu())  # 若有流式输出器，先把起始 prompt 推给它
        for _ in range(max_new_tokens):          # 最多生成 max_new_tokens 个 token
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0  # 已缓存的长度（决定本步只需喂入新 token）
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)  # 前向：只算未缓存部分
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None  # mask 末尾补 1（新 token 可见）
            logits = outputs.logits[:, -1, :] / temperature  # 取最后一个位置的 logits 并按温度缩放（温度越高越随机）
            if repetition_penalty != 1.0:         # 重复惩罚：对已出现过的 token 降低概率
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)  # 正分除、负分乘，均压低
            if top_k > 0:                         # top-k 截断：只保留概率最高的 k 个，其余置 -inf
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:                       # top-p（核）采样：保留累计概率达 p 的最小集合
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 概率从高到低排序
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p  # 累计概率超过 p 的部分标记为丢弃
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0  # 右移一位并保证至少保留概率最高的 1 个
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')  # 把丢弃位置映射回原顺序并置 -inf
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)  # 采样：按概率抽样 / 贪心取最大
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)  # 已结束的序列后续一律填 eos
            input_ids = torch.cat([input_ids, next_token], dim=-1)  # 把新 token 接到序列末尾
            past_key_values = outputs.past_key_values if use_cache else None  # 更新 KV 缓存
            if streamer: streamer.put(next_token.cpu())  # 把新 token 推给流式输出器
            if eos_token_id is not None:          # 检查是否所有序列都生成了 eos
                finished |= next_token.squeeze(-1).eq(eos_token_id)  # 标记新结束的序列
                if finished.all(): break          # 全部结束则提前停止
        if streamer: streamer.end()               # 通知流式输出器结束
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}  # 需要时连同 KV 一起返回（供续接生成）
        return input_ids                          # 默认只返回完整的 token 序列（含 prompt）
