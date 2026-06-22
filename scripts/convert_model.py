import os                       # 路径拼接、判断 transformers 版本时用
import sys                       # 修改模块搜索路径
import json                      # 读写 config / tokenizer_config / chat_template

__package__ = "scripts"         # 显式声明本文件所属包，便于以包方式被引用
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 把项目根加入搜索路径，使 `from model...` 可用
import torch                     # 加载/保存权重、精度转换
import transformers              # 读取其版本号以做新旧版本兼容处理
import warnings                  # 屏蔽告警
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM  # 分词器、通用模型、以及 Qwen3/Qwen3-MoE 的配置与模型类
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # MiniMind 原生配置与模型
from model.model_lora import apply_lora, merge_lora  # LoRA 挂载与合并工具

warnings.filterwarnings('ignore', category=UserWarning)  # 只屏蔽 UserWarning，保持输出干净

def convert_torch2transformers_minimind(torch_path, transformers_path, dtype=torch.float16):  # 把原生 .pth 转成「MiniMind 自定义结构」的 transformers 格式（保留自定义建模代码）
    MiniMindConfig.register_for_auto_class()  # 注册配置类，使保存后可被 AutoConfig 自动识别
    MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")  # 注册模型类，使可被 AutoModelForCausalLM 自动加载
    lm_model = MiniMindForCausalLM(lm_config)  # 用全局 lm_config 构造模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备
    state_dict = torch.load(torch_path, map_location=device)  # 读取原生权重
    lm_model.load_state_dict(state_dict, strict=False)  # 宽松加载（允许少量键不匹配，如非持久 buffer）
    lm_model = lm_model.to(dtype)  # 转换模型权重精度
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)  # 统计可训练参数量
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')  # 打印参数量（百万/十亿）
    lm_model.save_pretrained(transformers_path, safe_serialization=False)  # 保存为 transformers 格式（不使用 safetensors，保留 .bin）
    tokenizer = AutoTokenizer.from_pretrained('../model/')  # 加载项目自带分词器
    tokenizer.save_pretrained(transformers_path)  # 一并保存分词器到目标目录
    # ======= transformers-5.0的兼容低版本写法 =======
    if int(transformers.__version__.split('.')[0]) >= 5:  # transformers 5.x 的字段有变化，需要回填/清理
        tokenizer_config_path, config_path = os.path.join(transformers_path, "tokenizer_config.json"), os.path.join(transformers_path, "config.json")  # 两个配置文件路径
        json.dump({**json.load(open(tokenizer_config_path, 'r', encoding='utf-8')), "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}}, open(tokenizer_config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)  # 强制 tokenizer_class 并补 extra_special_tokens，兼容旧版加载
        config = json.load(open(config_path, 'r', encoding='utf-8'))  # 读 config.json
        config['rope_theta'] = lm_config.rope_theta; config['rope_scaling'] = None; del config['rope_parameters']  # 回填 rope_theta、清 rope_scaling、删掉新版独有的 rope_parameters
        json.dump(config, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)  # 写回 config.json
    print(f"模型已保存为 Transformers-MiniMind 格式: {transformers_path}")  # 完成提示


# QwenForCausalLM/LlamaForCausalLM结构兼容生态
def convert_torch2transformers(torch_path, transformers_path, dtype=torch.float16):  # 把原生 .pth 转成「Qwen3 标准结构」格式，便于接入 vLLM/ollama 等生态
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
    state_dict = torch.load(torch_path, map_location=device)  # 读取原生权重
    common_config = {            # MiniMind 与 Qwen3 共有的结构超参映射
        "vocab_size": lm_config.vocab_size,
        "hidden_size": lm_config.hidden_size,
        "intermediate_size": lm_config.intermediate_size,
        "num_hidden_layers": lm_config.num_hidden_layers,
        "num_attention_heads": lm_config.num_attention_heads,
        "num_key_value_heads": lm_config.num_key_value_heads,
        "head_dim": lm_config.hidden_size // lm_config.num_attention_heads,
        "max_position_embeddings": lm_config.max_position_embeddings,
        "rms_norm_eps": lm_config.rms_norm_eps,
        "rope_theta": lm_config.rope_theta,
        "tie_word_embeddings": lm_config.tie_word_embeddings
    }
    if not lm_config.use_moe:    # 稠密模型 → 映射到 Qwen3
        qwen_config = Qwen3Config(
            **common_config,
            use_sliding_window=False,  # MiniMind 不用滑动窗口注意力
            sliding_window=None
        )
        qwen_model = Qwen3ForCausalLM(qwen_config)  # 构造 Qwen3 模型
    else:                        # MoE 模型 → 映射到 Qwen3-MoE
        qwen_config = Qwen3MoeConfig(
            **common_config,
            num_experts=lm_config.num_experts,              # 专家数
            num_experts_per_tok=lm_config.num_experts_per_tok,  # 每 token 激活专家数
            moe_intermediate_size=lm_config.moe_intermediate_size,  # 专家中间维度
            norm_topk_prob=lm_config.norm_topk_prob         # 是否归一化 top-k 权重
        )
        qwen_model = Qwen3MoeForCausalLM(qwen_config)  # 构造 Qwen3-MoE 模型
        # ======= transformers-5.0的兼容低版本写法 =======
        if int(transformers.__version__.split('.')[0]) >= 5:  # 新版把各专家权重「打包」成单一大张量，需要重排
            new_sd = {k: v for k, v in state_dict.items() if 'experts.' not in k or 'gate.weight' in k}  # 先保留所有非专家权重（专家的 gate 路由权重除外要留）
            for l in range(lm_config.num_hidden_layers):  # 逐层把分散的专家权重堆叠合并
                p = f'model.layers.{l}.mlp.experts'  # 该层专家权重的键前缀
                new_sd[f'{p}.gate_up_proj'] = torch.cat([torch.stack([state_dict[f'{p}.{e}.gate_proj.weight'] for e in range(lm_config.num_experts)]), torch.stack([state_dict[f'{p}.{e}.up_proj.weight'] for e in range(lm_config.num_experts)])], dim=1)  # 把各专家的 gate 与 up 堆叠后在中间维拼接成 gate_up 大张量
                new_sd[f'{p}.down_proj'] = torch.stack([state_dict[f'{p}.{e}.down_proj.weight'] for e in range(lm_config.num_experts)])  # 把各专家的 down 堆叠成一个大张量
            state_dict = new_sd  # 用重排后的权重替换

    qwen_model.load_state_dict(state_dict, strict=True)  # 严格加载（键必须完全对应）
    qwen_model = qwen_model.to(dtype)  # 转换模型权重精度
    qwen_model.save_pretrained(transformers_path)  # 保存为 transformers 格式
    model_params = sum(p.numel() for p in qwen_model.parameters() if p.requires_grad)  # 统计参数量
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')  # 打印
    tokenizer = AutoTokenizer.from_pretrained('../model/')  # 加载分词器
    tokenizer.save_pretrained(transformers_path)  # 一并保存

    # ======= transformers-5.0的兼容低版本写法 =======
    if int(transformers.__version__.split('.')[0]) >= 5:  # 同上：5.x 下回填/清理配置字段
        tokenizer_config_path, config_path = os.path.join(transformers_path, "tokenizer_config.json"), os.path.join(transformers_path, "config.json")
        json.dump({**json.load(open(tokenizer_config_path, 'r', encoding='utf-8')), "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}}, open(tokenizer_config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        config = json.load(open(config_path, 'r', encoding='utf-8'))
        config['rope_theta'] = lm_config.rope_theta; config['rope_scaling'] = None; del config['rope_parameters']
        json.dump(config, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"模型已保存为 Transformers 格式: {transformers_path}")  # 完成提示


def convert_transformers2torch(transformers_path, torch_path):  # 反向转换：transformers 格式 → 原生 .pth
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)  # 加载 transformers 模型（允许自定义代码）
    torch.save({k: v.cpu().half() for k, v in model.state_dict().items()}, torch_path)  # 取出权重转 fp16/CPU 后保存为 .pth
    print(f"模型已保存为 PyTorch 格式: {torch_path}")  # 完成提示


def convert_merge_base_lora(base_torch_path, lora_path, merged_torch_path):  # 把 LoRA 合并进基模，导出合并后的原生 .pth
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
    lm_model = MiniMindForCausalLM(lm_config).to(device)  # 构造基模并搬到设备
    state_dict = torch.load(base_torch_path, map_location=device)  # 读取基模权重
    lm_model.load_state_dict(state_dict, strict=False)  # 宽松加载基模权重
    apply_lora(lm_model)  # 给基模挂上 LoRA 结构
    merge_lora(lm_model, lora_path, merged_torch_path)  # 加载并把 LoRA 增量合并进权重后保存
    print(f"LoRA 已合并并保存为基模结构 PyTorch 格式: {merged_torch_path}")  # 完成提示


def convert_jinja_to_json(jinja_path):  # 把 .jinja 聊天模板转成可放进 tokenizer_config.json 的 JSON 字符串
    with open(jinja_path, 'r') as f: template = f.read()  # 读取模板文本
    escaped = json.dumps(template)  # 用 json.dumps 做转义（处理引号/换行等）
    print(f'"chat_template": {escaped}')  # 打印成 "chat_template": "..." 形式，方便复制


def convert_json_to_jinja(json_file_path, output_path):  # 反向：从 tokenizer_config.json 抽出 chat_template 写成 .jinja 文件
    with open(json_file_path, 'r') as f: config = json.load(f)  # 读取 json 配置
    template = config['chat_template']  # 取出模板字段
    with open(output_path, 'w') as f: f.write(template)  # 写成独立的 jinja 文件
    print(f"模板已保存为 jinja 文件: {output_path}")  # 完成提示


if __name__ == '__main__':       # 作为脚本直接运行时执行下面的转换流程
    lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, max_seq_len=8192, use_moe=False)  # 全局配置（需与待转换权重一致）

    # convert torch to transformers
    torch_path = f"../out/full_sft_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"  # 待转换的原生权重路径
    transformers_path = '../minimind-3'  # 输出目录
    convert_torch2transformers(torch_path, transformers_path)  # 执行：原生 → Qwen3 标准格式

    # # merge lora
    # base_torch_path = f"../out/full_sft_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    # lora_path = f"../out/lora_identity_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    # merged_torch_path = f"../out/merge_identity_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    # convert_merge_base_lora(base_torch_path, lora_path, merged_torch_path)

    # convert_transformers2torch(transformers_path, torch_path)
    # convert_json_to_jinja('../model/tokenizer_config.json', '../model/chat_template.jinja')
    # convert_jinja_to_json('../model/chat_template.jinja')
