import os                            # 路径、目录
import sys                           # 模块搜索路径

__package__ = "trainer"             # 显式声明所属包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 项目根加入搜索路径

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse                      # 命令行参数
import math                          # 计算总优化步数（调度器 T_max）
import re                            # 解析 prompt 里的对话、重复惩罚分词
import warnings                      # 屏蔽告警
import torch                         # 训练框架
import torch.distributed as dist     # 分布式
import torch.nn.functional as F      # log_softmax 等
from transformers import AutoTokenizer  # 分词器
from contextlib import nullcontext   # CPU 上替代 autocast
from torch import optim, nn          # 优化器与模块（critic 的 value_head）
from torch.nn.parallel import DistributedDataParallel  # 多卡
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载/采样
from torch.nn.utils import clip_grad_norm_  # 梯度裁剪
from torch.optim.lr_scheduler import CosineAnnealingLR  # 余弦退火调度器
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # 模型配置与因果语言模型（critic 继承它）
from dataset.lm_dataset import RLAIFDataset  # 在线 RL 数据集（只给 prompt）
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel  # 工具集 + 奖励模型
from trainer.rollout_engine import create_rollout_engine  # 采样引擎工厂

warnings.filterwarnings('ignore')


def rep_penalty(text, n=3, cap=0.5):  # 重复惩罚：统计 n-gram 重复比例，重复越多惩罚越大（上限 cap）
    toks = re.findall(r"\w+|[^\w\s]", text.lower())  # 粗分词（单词或单个标点）
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]  # 所有 n-gram
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0  # (总数-去重数)/总数 衡量重复度


# 自定义的Critic模型，继承自MiniMindLM
class CriticModel(MiniMindForCausalLM):  # 价值网络：复用 MiniMind 骨架，但输出每个位置的“状态价值”而非词表 logits
    def __init__(self, params):
        super().__init__(params)
        # 替换lm_head为输出单一价值的线性层
        self.value_head = nn.Linear(params.hidden_size, 1)  # 价值头：hidden → 1

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # 使用基础模型获取隐藏状态
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)  # 走骨架
        hidden_states = self.model.norm(outputs[0])  # 末端归一化后的隐藏态
        # 使用value_head获取价值估计
        values = self.value_head(hidden_states).squeeze(-1)  # 每个 token 位置一个标量价值 [B, L]
        return values


def calculate_rewards(prompts, responses, reward_model):  # 计算每条回复的总奖励（规则奖励 + 奖励模型打分）
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():            # 奖励计算不需要梯度
        reward_model_scores = []
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"  # 从 prompt 文本还原对话
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]
            answer = response
            rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5  # 长度适中给正奖励，否则惩罚
            if '</think>' in response:  # 含思考段的额外规则奖励
                thinking_content, answer_content = response.split('</think>', 1)
                rewards[i] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5  # 思考长度适中
                rewards[i] += 0.25 if response.count('</think>') == 1 else -0.25  # 思考标签恰好一个
                answer = answer_content.strip()  # 真正的回答部分（去掉思考）
            rewards[i] -= rep_penalty(answer)  # 减去重复惩罚

            score = reward_model.get_score(messages, answer)  # 奖励模型对(对话, 回答)打分
            reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores  # 规则奖励 + 模型打分

    return rewards


def ppo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, start_step=0, wandb=None, use_sglang=False):  # PPO 训练一个 epoch
    actor_model.train()              # 策略网络（actor）训练模式
    critic_model.train()             # 价值网络（critic）训练模式
    grad_accum_step = 0              # 梯度累积计数

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]  # list[str], length B
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_len,
                        padding_side="left").to(args.device)  # input_ids: [B, P], attention_mask: [B, P]  # 左侧 padding，便于续写对齐

        rollout_result = rollout_engine.rollout(  # —— 采样阶段：用当前策略生成回复 ——
            prompt_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            num_generations=1,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        gen_out = rollout_result.output_ids  # 完整序列 [B, P+R]
        completion_ids = rollout_result.completion_ids  # 仅续写部分
        prompt_lens = rollout_result.prompt_lens.to(args.device)  # 各 prompt 长度
        responses_text = rollout_result.completions  # 续写文本
        old_resp_logp = rollout_result.per_token_logps.to(args.device)  # 采样时的旧策略 logprob（PPO 比值的分母）
        rewards = calculate_rewards(prompts, responses_text, reward_model)  # [B]  # 每条回复的奖励

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:  # 调试：打印采样样本
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-'*100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                Logger(f"[DEBUG] prompt_len={prompt_lens[i].item()}, response_len={len(responses_text[i])}")
                Logger(f"{'=' * 28} [DEBUG] sample[{i}] RESPONSE_BEGIN {'=' * 28}")
                Logger(responses_text[i])
                Logger(f"{'=' * 29} [DEBUG] sample[{i}] RESPONSE_END {'=' * 29}")
                Logger(f"[DEBUG] reward={rewards[i].item():.4f}")
                Logger('='*100)

        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]  # 整条序列有效位置
        labels = gen_out[:, 1:].clone()  # [B, P+R-1]  # 下一 token 标签（错位）
        B = len(prompts)
        resp_labels = completion_ids
        resp_idx = torch.arange(resp_labels.size(1), device=gen_out.device).unsqueeze(0)  # 续写内部位置索引 [1, R]
        logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx  # 续写每个 token 在完整序列 logits 里对应的位置
        resp_pad_mask = rollout_result.completion_mask.to(args.device).bool()  # 续写有效位置（非 pad）
        resp_lengths = resp_pad_mask.sum(dim=1); valid_resp = resp_lengths > 0; eos_mask = resp_labels.eq(tokenizer.eos_token_id) & resp_pad_mask  # 各回复长度、是否非空、eos 位置
        has_eos = eos_mask.any(dim=1); eos_pos = torch.argmax(eos_mask.int(), dim=1)  # 是否含 eos、首个 eos 下标
        resp_lengths = torch.where(has_eos, eos_pos + 1, resp_lengths).long().clamp(min=1)  # 真实长度截到 eos（含）
        resp_policy_mask = ((resp_idx < resp_lengths.unsqueeze(1)) & resp_pad_mask).float()  # 策略损失的有效掩码
        resp_value_mask = resp_policy_mask.clone()  # 价值损失掩码（同上）

        with torch.no_grad():  # Rollout阶段只需推理获取old_logp和old_values，切断梯度省显存
            critic_for_rollout = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model
            values_seq = critic_for_rollout(input_ids=gen_out, attention_mask=full_mask)  # critic 给出每个位置的价值
            old_resp_values = values_seq.gather(1, logp_pos) * resp_value_mask  # 取续写位置的旧价值

            ref_resp_logp = F.log_softmax(ref_model(input_ids=gen_out, attention_mask=full_mask).logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)  # 参考模型对续写 token 的 logprob（KL 惩罚用）
            token_rewards = torch.zeros_like(old_resp_logp)  # 逐 token 奖励（稀疏：只在末尾给）
            last_idx = resp_lengths - 1  # [B]  # 每条回复最后一个 token 位置
            token_rewards[torch.arange(B, device=args.device)[valid_resp], last_idx[valid_resp]] += rewards[valid_resp]  # 末尾加外部奖励

            gen_len = old_resp_values.size(1); lastgaelam = torch.zeros(B, device=args.device); advs_rev = []  # —— GAE 计算优势 ——
            for t in reversed(range(gen_len)):  # 从后往前递推广义优势估计
                nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0  # 下一步价值（末步为 0）
                delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]  # TD 残差 δ
                lastgaelam = delta + args.gamma * args.lam * lastgaelam  # GAE 递推：A_t = δ_t + γλ A_{t+1}
                advs_rev.append(lastgaelam)
            advantages = torch.stack(advs_rev[::-1], dim=1)  # [B, R]  # 翻转回正序
            returns = advantages + old_resp_values  # [B, R]  # 回报 = 优势 + 价值（critic 的回归目标）

            adv_mean = (advantages * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)  # 优势均值（仅有效位置）
            adv_var = ((advantages - adv_mean) ** 2 * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)  # 优势方差
            advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) * resp_policy_mask  # 优势标准化（稳定训练）

        mb_size = max(1, min(args.mini_batch_size, B))  # PPO minibatch 大小
        stop_ppo = False             # 早停标志（KL 过大时）
        policy_loss_sum = 0.0        # 以下为日志累加器
        value_loss_sum = 0.0
        kl_sum = 0.0
        kl_ref_sum = 0.0
        clipfrac_sum = 0.0
        aux_loss_sum = 0.0
        log_count = 0
        actor_unwrapped = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model  # 解 DDP
        critic_unwrapped = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model
        for ppo_epoch in range(args.ppo_update_iters):  # 同一批 rollout 数据重复更新若干次（PPO 特点）
            if stop_ppo:
                break
            b_inds = torch.randperm(B, device=args.device)  # 打乱
            for i in range(0, B, mb_size):  # 逐 minibatch
                inds = b_inds[i:i + mb_size]

                mb_values_seq = critic_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])  # critic 当前价值
                mb_resp_values = mb_values_seq.gather(1, logp_pos[inds])  # 取续写位置

                with autocast_ctx:
                    res = actor_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])  # actor 前向
                    aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)  # MoE 辅助损失

                mb_resp_logp = F.log_softmax(res.logits[:, :-1], dim=-1).gather(2, labels[inds].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos[inds])  # 当前策略对续写 token 的 logprob

                log_ratio = mb_resp_logp - old_resp_logp[inds]  # log(π_new/π_old)
                approx_kl = (0.5 * (log_ratio ** 2) * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)  # 近似 KL（监控用）

                # 同步各卡的 approx_kl，防止某卡 break 而其它卡继续导致 DDP 死锁
                approx_kl_val = approx_kl.detach().clone()
                if dist.is_initialized():
                    dist.all_reduce(approx_kl_val, op=dist.ReduceOp.AVG)  # 各卡取平均，保证早停决策一致

                if approx_kl_val > args.early_stop_kl:  # KL 过大 → 触发早停
                    stop_ppo = True

                ratio = torch.exp(log_ratio)  # 重要性采样比值 π_new/π_old
                clipfrac = ((((ratio - 1.0).abs() > args.clip_epsilon).float() * resp_policy_mask[inds]).sum()
                            / resp_policy_mask[inds].sum().clamp(min=1))  # 被裁剪比例（监控）
                kl_ref_penalty = ((torch.exp(ref_resp_logp[inds] - mb_resp_logp) - (ref_resp_logp[inds] - mb_resp_logp) - 1.0)
                                  * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)  # 相对参考模型的 KL 惩罚（k3 估计）
                policy_loss = ((torch.max(-advantages[inds] * ratio,
                                          -advantages[inds] * torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon))
                               * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                               + args.kl_coef * kl_ref_penalty)  # PPO-Clip 策略损失（取裁剪/未裁剪较大者）+ KL 惩罚
                value_loss = 0.5 * (torch.max((mb_resp_values - returns[inds]) ** 2,
                                              (torch.clamp(mb_resp_values, old_resp_values[inds] - args.cliprange_value,
                                                           old_resp_values[inds] + args.cliprange_value) - returns[inds]) ** 2)
                                    * resp_value_mask[inds]).sum() / resp_value_mask[inds].sum().clamp(min=1)  # 价值损失（同样做裁剪，取较大者）

                kl = approx_kl_val
                kl_ref = kl_ref_penalty.detach()

                # 早停时必须保证 forward-backward 闭环，故只截断 loss 不中断 DDP 通信
                if stop_ppo:
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) * 0.0  # 乘 0：仍走 backward 以保持各卡通信同步，但不更新
                else:
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) / args.accumulation_steps  # 总损失（策略 + 价值 + 辅助）

                loss.backward()      # 反向（actor 与 critic 共享这次 backward）

                policy_loss_sum += policy_loss.item()  # 累加日志
                value_loss_sum += value_loss.item()
                kl_sum += kl.item()
                kl_ref_sum += kl_ref.item()
                clipfrac_sum += clipfrac.item()
                aux_loss_sum += aux_loss.item()
                log_count += 1

                grad_accum_step += 1

                if grad_accum_step % args.accumulation_steps == 0:  # 累积满则更新
                    clip_grad_norm_(actor_model.parameters(), args.grad_clip)
                    clip_grad_norm_(critic_model.parameters(), args.grad_clip)
                    actor_optimizer.step()
                    critic_optimizer.step()
                    actor_scheduler.step()
                    critic_scheduler.step()
                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()

        if grad_accum_step % args.accumulation_steps != 0:  # 处理尾部剩余梯度
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(actor_model)  # 把更新后的策略同步给采样引擎

        if is_main_process():        # 主进程记录日志
            critic_loss_val = value_loss_sum / max(log_count, 1)
            reward_val = rewards.mean().item()
            approx_kl_val = kl_sum / max(log_count, 1)
            kl_ref_val = kl_ref_sum / max(log_count, 1)
            clipfrac_val = clipfrac_sum / max(log_count, 1)
            avg_len_val = resp_lengths.float().mean().item()
            actor_lr, critic_lr = actor_optimizer.param_groups[0]['lr'], critic_optimizer.param_groups[0]['lr']

            if wandb is not None:
                wandb.log({
                    "reward": reward_val,
                    "kl_ref": kl_ref_val,
                    "approx_kl": approx_kl_val,
                    "clipfrac": clipfrac_val,
                    "critic_loss": critic_loss_val,
                    "avg_response_len": avg_len_val,
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr,
                })

            Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                   f"Reward: {reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, Approx KL: {approx_kl_val:.4f}, "
                   f"ClipFrac: {clipfrac_val:.4f}, Critic Loss: {critic_loss_val:.4f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.8f}, Critic LR: {critic_lr:.8f}")

        if (step % args.save_interval == 0 or step == iters) and is_main_process():  # 保存
            actor_model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            actor_state = raw_actor.state_dict()
            torch.save({k: v.half().cpu() for k, v in actor_state.items()}, ckp)  # 只保存 actor 权重供推理

            # 使用 lm_checkpoint 保存完整状态（包括 critic）
            lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer,
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                         scheduler=actor_scheduler, critic_model=critic_model,
                         critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)  # 续训断点含 critic 全套
            actor_model.train()
            del actor_state

        del enc, gen_out, completion_ids, responses_text, rewards, full_mask, values_seq, advantages  # 释放显存
        del labels, resp_labels, resp_idx, resp_pad_mask, valid_resp, eos_mask, has_eos, eos_pos, resp_lengths, resp_policy_mask, resp_value_mask, old_resp_logp, ref_resp_logp
        del kl, kl_ref, policy_loss, value_loss, loss, token_rewards, returns, old_resp_values, prompt_lens, logp_pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")  # 近端策略优化
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=5e-7, help="Critic学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO裁剪参数")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")
    parser.add_argument("--gamma", type=float, default=1.0, help="GAE折扣因子")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda参数")
    parser.add_argument("--cliprange_value", type=float, default=0.2, help="Value function裁剪范围")
    parser.add_argument("--ppo_update_iters", type=int, default=2, help="同一批rollout重复更新次数")
    parser.add_argument("--early_stop_kl", type=float, default=0.25, help="PPO early stop 的 KL 阈值")
    parser.add_argument("--mini_batch_size", type=int, default=2, help="PPO每次更新的minibatch大小")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_ppo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 初始化模型和数据 ==========
    base_weight = args.from_weight
    # Actor模型
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)  # 策略网络
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)  # 参考网络（冻结，算 KL）
    ref_model = ref_model.eval().requires_grad_(False)
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    critic_model = CriticModel(lm_config)  # 价值网络（用同一份 SFT 权重初始化骨架）
    critic_model.load_state_dict(state_dict, strict=False)  # 宽松加载（value_head 是新加的，原权重里没有）
    critic_model = critic_model.to(args.device)
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)  # 奖励模型
    # Rollout引擎
    rollout_engine = create_rollout_engine(  # 采样引擎（torch 或 sglang）
        engine_type=args.rollout_engine,
        policy_model=actor_model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len), thinking_ratio=args.thinking_ratio)  # 只给 prompt 的数据集
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)  # 仅用于统计总步数
    iters = len(loader_for_count)
    mb_factor = max(1, math.ceil(args.batch_size / args.mini_batch_size))  # 每个 step 内的 minibatch 数
    total_optimizer_steps = math.ceil(iters * args.epochs * args.ppo_update_iters * mb_factor / args.accumulation_steps)  # 调度器总步数
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)  # 余弦退火
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)

    start_epoch, start_step = 0, 0
    if ckp_data:                     # 续训：恢复 actor/critic 及其优化器、调度器
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        actor_model = torch.compile(actor_model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(actor_model)  # 编译后同步给采样引擎
    if dist.is_initialized():
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
    rollout_engine.update_policy(actor_model)  # 初始同步策略

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            ppo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, start_step, wandb, use_sglang = (args.rollout_engine == "sglang"))
        else:
            ppo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, 0, wandb, use_sglang = (args.rollout_engine == "sglang"))

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
