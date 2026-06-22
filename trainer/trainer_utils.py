"""
训练工具函数集合
"""
import os                            # 环境变量、路径、文件替换
import sys                           # 修改模块搜索路径
__package__ = "trainer"             # 显式声明所属包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 把项目根加入搜索路径，便于 `from model...`
import random                        # 设随机种子
import math                          # 学习率余弦调度
import numpy as np                   # 设随机种子
import torch                         # 张量与训练
import torch.distributed as dist     # 分布式训练（DDP）相关
from torch.nn.parallel import DistributedDataParallel  # DDP 封装类型判断/解包
from torch.utils.data import Sampler  # 自定义采样器基类
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification  # 分词器、通用模型、序列分类（奖励模型）
from model.model_minimind import MiniMindForCausalLM  # MiniMind 因果语言模型

def get_model_params(model, config):  # 统计并打印模型参数量（对 MoE 额外区分“总量”与“激活量”）
    total = sum(p.numel() for p in model.parameters()) / 1e6  # 总参数量（百万）
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))  # 路由专家数（无则 0）
    n_active = getattr(config, 'num_experts_per_tok', 0)  # 每 token 激活的专家数
    n_shared = getattr(config, 'n_shared_experts', 0)  # 共享专家数
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6  # 单个路由专家的参数量
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6  # 单个共享专家的参数量
    base = total - (expert * n_routed) - (shared_expert * n_shared)  # 去掉所有专家后的“骨架”参数量
    active = base + (expert * n_active) + (shared_expert * n_shared)  # 实际前向时激活的参数量（骨架 + 激活专家）
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')  # MoE：打印 总量-A激活量
    else: Logger(f'Model Params: {total:.2f}M')  # 稠密模型：只打印总量


def is_main_process():               # 是否主进程（非分布式时恒为 True；分布式时仅 rank0）
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):                 # 只在主进程打印，避免多卡重复刷屏
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):  # 余弦退火学习率：从 lr 衰减到 0.1*lr
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():         # 初始化分布式（DDP）环境
    if int(os.environ.get("RANK", -1)) == -1:  # 未设置 RANK → 非 DDP 模式
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")  # 用 NCCL 后端初始化进程组（GPU 通信）
    local_rank = int(os.environ["LOCAL_RANK"])  # 本机内的卡号
    torch.cuda.set_device(local_rank)  # 绑定当前进程到该卡
    return local_rank


def setup_seed(seed: int):           # 固定随机种子，保证可复现
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # cudnn 用确定性算法
    torch.backends.cudnn.benchmark = False     # 关闭自动算法搜索

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):  # 统一的“保存/加载”断点函数：传 model 则保存，不传则尝试加载
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    moe_path = '_moe' if lm_config.use_moe else ''  # MoE 后缀
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'  # 纯权重路径（用于推理）
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'  # 续训断点路径（含优化器等）

    if model is not None:            # —— 保存模式 ——
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model  # 解开 DDP 包装
        raw_model = getattr(raw_model, '_orig_mod', raw_model)  # 解开 torch.compile 包装
        state_dict = raw_model.state_dict()  # 取权重
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}  # 转 fp16/CPU，减小体积
        ckp_tmp = ckp_path + '.tmp'  # 先写临时文件，再原子替换，避免中断导致权重损坏
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)  # 原子替换
        wandb_id = None              # 记录 wandb 运行 id（便于续训时接上同一条曲线）
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {              # 续训断点：权重 + 优化器 + 进度 + 并行规模 + wandb id
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():  # 额外对象（如调度器、scaler、critic 等）也一并保存
            if value is not None:
                if hasattr(value, 'state_dict'):  # 有 state_dict 的对象保存其状态
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:                # 普通值直接保存
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'  # 同样先临时再原子替换
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data  # 释放内存
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):  # 有断点才加载
            ckp_data = torch.load(resume_path, map_location='cpu')  # 读到 CPU
            saved_ws = ckp_data.get('world_size', 1)  # 保存时的 GPU 数
            current_ws = dist.get_world_size() if dist.is_initialized() else 1  # 当前 GPU 数
            if saved_ws != current_ws:  # GPU 数变了 → 按比例换算已完成 step，保证续训进度一致
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None                  # 无断点返回 None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):  # 训练用的模型初始化：可从指定阶段权重热启动
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)  # 分词器
    model = MiniMindForCausalLM(lm_config)  # 新建模型

    if from_weight!= 'none':         # 'none' 表示从零开始；否则加载上一阶段权重
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'  # 上一阶段权重路径
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)  # 宽松加载（容忍结构微小差异）

    get_model_params(model, lm_config)  # 打印参数量
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')  # 打印可训练参数量
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):     # 自定义批采样器：可跳过前若干个 batch（用于断点续训时恢复到中断位置）
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler       # 底层样本采样器
        self.batch_size = batch_size
        self.skip_batches = skip_batches  # 要跳过的 batch 数

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:     # 攒够一个 batch 就 yield
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:  # 还没跳够 → 丢弃这个 batch
                    skipped += 1
                    batch = []
                    continue
                yield batch          # 正常产出
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:  # 处理最后不足一个 batch 的尾部
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size  # 总 batch 数（向上取整）
        return max(0, total_batches - self.skip_batches)  # 减去跳过的


class LMForRewardModel:              # 把一个语言模型包装成“奖励模型”：对(对话, 回复)打分（用于 RLHF/RLAIF）
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  # 奖励模型的分词器
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)  # 加载奖励模型
        self.model = self.model.to(device).eval()  # 评估模式
        self.device = device

    @torch.no_grad()                 # 打分不需要梯度
    def get_score(self, messages, response):  # 给一段对话 + 候选回复打分
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])  # 把历史拼成文本
        last_query = messages[-1]['content'] if messages else ""  # 最后一条用户问题
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query  # 组合上下文
        eval_messages = [            # 构造“用户问 + 助手答”给奖励模型评估
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)  # 调奖励模型打分
        return max(min(score, 3.0), -3.0)  # 把分数裁剪到 [-3, 3]，稳定训练
