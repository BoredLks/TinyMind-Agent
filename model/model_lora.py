# LoRA（Low-Rank Adaptation，低秩适配）实现。
# 核心思想：冻结原始大权重 W 不训练，只在它旁边并联一条“低秩旁路” ΔW = B·A（rank 远小于维度），
# 只训练这条旁路。这样微调所需的可训练参数量极小，却能让模型学到新任务/新风格。
# 本文件提供：LoRA 模块本身，以及把它挂到模型(apply)、从磁盘加载(load)、保存(save)、
# 永久合并回主干权重(merge) 的工具函数；trainer/train_lora.py 与各推理脚本会用到它们。
import torch                      # PyTorch 主包，提供张量、torch.load / torch.save 等
from torch import optim, nn       # optim：优化器命名空间（本文件未直接使用）；nn：神经网络层与模块基类


# 定义Lora网络结构
class LoRA(nn.Module):            # LoRA 旁路：用两个无偏置线性层 A、B 串联来表达低秩矩阵 B·A
    def __init__(self, in_features, out_features, rank):  # 维度与被适配的 Linear 对齐；rank 为低秩维度
        super().__init__()        # 初始化 nn.Module 基类（注册子模块/参数所必需）
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A：把输入从 in_features 压到 rank 维
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B：再从 rank 维放回 out_features 维
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)     # A 用小标准差高斯初始化，提供随机的下降方向
        # 矩阵B全0初始化
        self.B.weight.data.zero_()                          # B 初始化为 0 → B·A=0，保证“刚挂上 LoRA 时完全不改变原模型输出”

    def forward(self, x):         # 前向：x → A 降维 → B 升维
        return self.B(self.A(x))  # 仅返回低秩旁路的增量（原始层输出在 apply_lora 的新 forward 里相加）


def apply_lora(model, rank=16):   # 遍历模型，给“方阵”线性层挂上 LoRA，并改写它们的 forward
    for name, module in model.named_modules():  # 递归遍历所有子模块，name 是该模块的点分路径名
        # 只对权重为方阵（输入维==输出维）的 Linear 挂 LoRA，例如 MiniMind 里 hidden→hidden 的 q_proj / o_proj
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)  # 建同形状 LoRA 并搬到模型所在设备
            setattr(module, "lora", lora)            # 把 lora 作为该 Linear 的属性挂上，便于后续按层定位加载/保存
            original_forward = module.forward        # 备份该层原本的 forward（只走原始权重的那条计算路径）

            # 显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):  # 用默认参数把当前的 original_forward/lora “快照”进来，规避 Python 闭包的延迟绑定坑
                return layer1(x) + layer2(x)         # 新前向 = 原始输出 + LoRA 旁路增量

            module.forward = forward_with_lora       # 用带 LoRA 的前向替换掉原前向


def load_lora(model, path):       # 从磁盘加载训练好的 LoRA 权重，回填到各层的 .lora 子模块
    state_dict = torch.load(path, map_location=model.device)  # 读取权重字典并映射到模型所在设备
    # 去掉分布式训练(DDP)可能给键名加上的 'module.' 前缀，统一成无前缀形式
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():       # 再次遍历模型各层
        if hasattr(module, 'lora'):                  # 找到此前挂了 LoRA 的层
            # 从总字典中筛出属于本层的键，并去掉 '{name}.lora.' 前缀，得到 LoRA 子模块自身的 state_dict
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)  # 把这部分权重加载进该层的 LoRA 子模块


def save_lora(model, path):       # 仅保存 LoRA 旁路参数（不含主干权重），所以文件非常小
    raw_model = getattr(model, '_orig_mod', model)   # 若模型被 torch.compile 包裹，取出底层原始模型(_orig_mod)
    state_dict = {}                                  # 收集所有 LoRA 参数
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name  # 同样去掉可能的 'module.' 前缀
            # 以 '{层名}.lora.{参数名}' 为键，转 fp16 并搬到 CPU 后保存，进一步压缩体积
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)                     # 落盘保存


def merge_lora(model, lora_path, save_path):  # 把 LoRA 增量永久合并进主干权重，导出一个“即使不加载 LoRA 也带新能力”的完整模型
    load_lora(model, lora_path)                      # 先把 LoRA 权重加载回模型各层
    raw_model = getattr(model, '_orig_mod', model)   # 取原始模型
    # 先收集所有“非 LoRA”的参数（即主干权重与其它张量），统一转 fp16/CPU
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:  # 遍历每个主干 Linear
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()  # 先放入该层原始权重
            if hasattr(module, 'lora'):              # 若该层挂了 LoRA
                # 计算低秩增量矩阵 B·A 并叠加到原权重上：W' = W + B·A，完成数学上的等价合并
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    torch.save(state_dict, save_path)                # 导出合并后的完整权重文件
