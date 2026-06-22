import os                            # 路径、目录创建、环境变量
import sys                           # 修改模块搜索路径

__package__ = "trainer"             # 显式声明所属包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 项目根加入搜索路径，便于 `from model... / from dataset...`

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse                      # 命令行参数
import time                          # 计时与 ETA 估算
import warnings                      # 屏蔽告警
import torch                         # 训练主框架
import torch.distributed as dist     # 分布式（DDP）
from contextlib import nullcontext   # CPU 上用空上下文替代 autocast
from torch import optim, nn          # 优化器与神经网络模块
from torch.nn.parallel import DistributedDataParallel  # 多卡数据并行
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载与分布式采样
from model.model_minimind import MiniMindConfig  # 模型配置
from dataset.lm_dataset import PretrainDataset  # 预训练数据集
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler  # 训练工具集

warnings.filterwarnings('ignore')   # 忽略告警


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):  # 训练一个 epoch
    start_time = time.time()         # 记录起始时间（算速度/ETA）
    last_step = start_step           # 记录最后到达的 step（用于处理尾部不满累积步）
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):  # 遍历各 batch（step 从断点位置+1 起算）
        input_ids = input_ids.to(args.device)  # 数据搬到设备
        labels = labels.to(args.device)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)  # 按全局进度算余弦退火学习率
        for param_group in optimizer.param_groups:  # 写入优化器
            param_group['lr'] = lr

        with autocast_ctx:           # 混合精度前向
            res = model(input_ids, labels=labels)  # 前向得到 loss（next-token 交叉熵）
            loss = res.loss + res.aux_loss  # 主损失 + MoE 辅助损失（无 MoE 时辅助损失为 0）
            loss = loss / args.accumulation_steps  # 梯度累积：按累积步数缩放

        scaler.scale(loss).backward()  # 反向（GradScaler 处理 fp16 数值缩放）

        if step % args.accumulation_steps == 0:  # 累积满则更新一次参数
            scaler.unscale_(optimizer)  # 先反缩放梯度，才能正确裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪，防爆炸

            scaler.step(optimizer)   # 优化器更新
            scaler.update()          # 更新缩放因子

            optimizer.zero_grad(set_to_none=True)  # 清梯度（set_to_none 更省显存）

        if step % args.log_interval == 0 or step == iters:  # 按间隔打日志
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps  # 还原成未缩放的真实 loss
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0  # MoE 辅助损失
            current_logits_loss = current_loss - current_aux_loss  # 纯语言建模损失
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60  # 估算本 epoch 剩余分钟
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})  # 记录到可视化面板

        if (step % args.save_interval == 0 or step == iters) and is_main_process():  # 按间隔在主进程保存
            model.eval()             # 切评估模式再保存
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'  # 纯权重路径
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model  # 解 DDP
            raw_model = getattr(raw_model, '_orig_mod', raw_model)  # 解 compile
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)  # 保存 fp16/CPU 纯权重（供推理）
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')  # 另存续训断点
            model.train()            # 切回训练模式
            del state_dict           # 释放

        del input_ids, labels, res, loss  # 及时释放，缓解显存压力

    if last_step > start_step and last_step % args.accumulation_steps != 0:  # 处理尾部：最后一段不满累积步也更新一次
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":           # 脚本入口
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")  # 预训练参数
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()       # 解析参数

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()  # 初始化 DDP（非分布式返回 0）
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"  # 分布式时绑定本进程的卡
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))  # 各进程用不同种子（基种子 42 + rank）

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)  # 确保保存目录存在
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))  # 构造模型配置
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None  # 续训则尝试加载断点

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"  # 设备类型
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16  # 精度
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)  # CPU 不用 autocast

    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():  # 仅主进程开实验记录
        import swanlab as wandb       # 这里用 swanlab 充当 wandb 接口
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None  # 续训时接上原 run
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)  # 初始化模型（可热启动）
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)  # 预训练数据集
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None  # 分布式采样器
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))  # fp16 才需要梯度缩放
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)  # AdamW 优化器

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:                     # 有断点则恢复模型/优化器/缩放器/进度
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:        # 可选 torch.compile 加速
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():        # 分布式则包成 DDP
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):  # 逐 epoch
        train_sampler and train_sampler.set_epoch(epoch)  # 分布式下设 epoch 以正确打乱
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()  # 单机下用随机种子打乱索引
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0  # 仅断点所在 epoch 需要跳过已训 step
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)  # 可跳过前若干 batch 的采样器
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)  # 数据加载器
        if skip > 0:                 # 断点续训：跳过后接着训
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:                        # 正常从头训该 epoch
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()  # 销毁进程组
