import time                       # 计时（统计生成速度 tokens/s）
import argparse                   # 解析命令行参数
import random                     # 随机数（用于每轮随机种子）
import warnings                   # 屏蔽告警
import torch                      # PyTorch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer  # 分词器、通用模型加载器、流式输出器
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM        # MiniMind 配置与因果语言模型
from model.model_lora import *   # LoRA 工具（apply_lora / load_lora 等）
from trainer.trainer_utils import setup_seed, get_model_params              # 设随机种子、打印模型参数量
warnings.filterwarnings('ignore')  # 忽略所有告警，保持输出干净

def init_model(args):            # 根据命令行参数加载模型与分词器
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)  # 从指定目录加载分词器（含 chat_template）
    if 'model' in args.load_from:  # load_from 含 'model' → 用原生 torch 权重(.pth)方式加载
        model = MiniMindForCausalLM(MiniMindConfig(  # 用配置实例化 MiniMind 模型
            hidden_size=args.hidden_size,            # 隐藏层维度
            num_hidden_layers=args.num_hidden_layers,  # 层数
            use_moe=bool(args.use_moe),              # 是否 MoE
            inference_rope_scaling=args.inference_rope_scaling  # 是否启用 RoPE 外推
        ))
        moe_suffix = '_moe' if args.use_moe else ''  # MoE 权重文件名带 _moe 后缀
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'  # 拼出权重路径，如 ./out/full_sft_768.pth
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)  # 严格加载权重（键必须完全匹配）
        if args.lora_weight != 'None':  # 若指定了 LoRA 权重
            apply_lora(model)            # 先给模型挂上 LoRA 结构
            load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}.pth')  # 再加载 LoRA 权重
    else:                              # 否则按 transformers 标准格式从目录加载
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)  # 允许加载仓库自带的自定义模型代码
    get_model_params(model, model.config)  # 打印模型参数量信息
    return model.half().eval().to(args.device), tokenizer  # 转半精度、切到评估模式、搬到设备后返回

def main():                       # 命令行主流程：加载模型 → 读取/构造对话 → 流式生成
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")  # 创建参数解析器
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")  # 模型来源
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")  # .pth 权重所在目录
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")  # 权重前缀
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")  # 可选 LoRA
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")  # 必须与权重匹配
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")  # 必须与权重匹配
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")  # 是否 MoE
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")  # 长上下文外推开关
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")  # 单次生成上限
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")  # 采样温度
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus采样阈值（0-1）")  # 核采样阈值
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")  # 是否注入 <think> 思考段
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")  # 多轮上下文轮数
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")  # 是否打印速度
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")  # 自动选 GPU/CPU
    args = parser.parse_args()    # 解析得到 args

    prompts = [                   # 自动测试模式下依次使用的内置问题
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]

    conversation = []             # 对话历史（多轮时累积）
    model, tokenizer = init_model(args)  # 加载模型与分词器
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))  # 选择运行模式：0=跑内置问题，1=交互输入
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)  # 流式输出器：边生成边打印，跳过 prompt 与特殊符

    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')  # 0 用内置列表；1 用迭代器不断读取输入（空串结束）
    for prompt in prompt_iter:    # 逐个问题处理
        setup_seed(random.randint(0, 31415926))  # 每轮设一个随机种子，让采样结果可复现且各轮不同
        if input_mode == 0: print(f'💬: {prompt}')  # 自动模式下打印当前问题
        conversation = conversation[-args.historys:] if args.historys else []  # 按需保留最近若干轮历史（0 则清空）
        conversation.append({"role": "user", "content": prompt})  # 把当前问题加入对话
        if 'pretrain' in args.weight:  # 预训练权重没有对话模板，直接用 bos+prompt
            inputs = tokenizer.bos_token + prompt
        else:                      # 其余权重套用 chat 模板（按是否思考拼接）
            inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))

        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)  # 文本编码成张量并搬到设备

        print('🧠: ', end='')      # 打印模型回答前缀（不换行，后续流式追加）
        st = time.time()           # 记录开始时间
        generated_ids = model.generate(  # 调用自定义 generate 自回归生成
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],  # 输入与注意力掩码
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,  # 生成长度、采样、流式
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,  # pad/eos（pad 经 **kwargs 透传被忽略，eos 用于停止）
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1  # 核采样阈值、温度、重复惩罚
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)  # 只解码新生成部分（去掉 prompt）
        conversation.append({"role": "assistant", "content": response})  # 把回答写回对话历史（供多轮）
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])  # 本轮生成的 token 数
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')  # 打印速度或仅换行

if __name__ == "__main__":        # 作为脚本直接运行时
    main()                        # 启动主流程
