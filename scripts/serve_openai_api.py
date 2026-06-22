import argparse                 # 解析命令行参数
import json                      # 序列化流式/非流式响应、解析工具调用
import re                        # 正则：解析 <think> 思考段与 <tool_call> 工具调用
import os                        # 路径处理
import sys                       # 修改模块搜索路径

__package__ = "scripts"         # 显式声明所属包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 把项目根加入搜索路径，便于 `from model...`
import time                      # 生成响应 id/created 时间戳
import torch                     # 张量与推理
import warnings                  # 屏蔽告警
import uvicorn                   # ASGI 服务器，用于跑 FastAPI 应用

from threading import Thread     # 用子线程跑生成，使主线程能从队列流式取词
from queue import Queue          # 线程安全队列，作为生成线程→主线程的桥梁
from fastapi import FastAPI, HTTPException  # Web 框架与 HTTP 异常
from fastapi.responses import StreamingResponse  # 流式响应（SSE）
from pydantic import BaseModel   # 请求体的数据校验模型
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer  # 分词器、通用模型、流式器基类
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # MiniMind 配置与模型
from model.model_lora import apply_lora, load_lora  # LoRA 挂载与加载

warnings.filterwarnings('ignore')  # 忽略所有告警

app = FastAPI()                  # 创建 FastAPI 应用实例（路由挂在它上面）


def init_model(args):            # 加载模型与分词器（与 eval_llm 的逻辑一致，路径相对 scripts/ 目录）
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)  # 加载分词器
    if 'model' in args.load_from:  # 原生 .pth 权重方式
        moe_suffix = '_moe' if args.use_moe else ''  # MoE 后缀
        ckp = f'../{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'  # 权重路径（注意 ../，因为从 scripts/ 运行）
        model = MiniMindForCausalLM(MiniMindConfig(  # 构造模型
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            max_seq_len=args.max_seq_len,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        model.load_state_dict(torch.load(ckp, map_location=device), strict=True)  # 严格加载权重
        if args.lora_weight != 'None':  # 可选 LoRA
            apply_lora(model)            # 挂 LoRA 结构
            load_lora(model, f'../{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')  # 加载 LoRA 权重
    else:                              # transformers 标准格式
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)  # 从目录加载
    print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')  # 打印参数量
    return model.half().eval().to(device), tokenizer  # 半精度、评估模式、搬到设备后返回


class ChatRequest(BaseModel):    # /v1/chat/completions 的请求体结构（OpenAI 兼容字段）
    model: str                   # 模型名
    messages: list               # 对话消息列表
    temperature: float = 0.7     # 采样温度
    top_p: float = 0.92          # 核采样阈值
    max_tokens: int = 8192       # 最大生成长度
    stream: bool = True          # 是否流式
    tools: list = []             # 可用工具列表（function calling）
    open_thinking: bool = False  # 是否开启思考
    chat_template_kwargs: dict = None  # 透传给 chat 模板的额外参数（另一种开启思考的方式）

    def get_open_thinking(self) -> bool:
        """兼容多种方式开启 thinking"""
        if self.open_thinking:   # 直接置位
            return True
        if self.chat_template_kwargs:  # 或通过 chat_template_kwargs 传入
            return self.chat_template_kwargs.get('open_thinking', False) or \
                   self.chat_template_kwargs.get('enable_thinking', False)  # 兼容两种键名
        return False             # 默认不开启


class CustomStreamer(TextStreamer):  # 自定义流式器：把生成的文本片段塞进队列，供 HTTP 端流式吐出
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)  # 跳过 prompt 与特殊符
        self.queue = queue       # 输出队列
        self.tokenizer = tokenizer

    def on_finalized_text(self, text: str, stream_end: bool = False):  # 每当有一段文本就绪时被回调
        self.queue.put(text)     # 把文本片段放入队列
        if stream_end:           # 生成结束
            self.queue.put(None)  # 放入 None 作为结束哨兵


def parse_response(text):        # 从完整生成文本里拆出：正文、思考内容、工具调用
    reasoning_content = None     # 思考内容（默认无）
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)  # 匹配成对的 <think>..</think>
    if think_match:              # 有成对思考段
        reasoning_content = think_match.group(1).strip()  # 取思考内容
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)  # 从正文中删掉思考段
    elif '</think>' in text:     # 只有结束标签（思考被截断/无开始标签）
        parts = text.split('</think>', 1)  # 以 </think> 切一刀
        reasoning_content = parts[0].strip()  # 前半作为思考
        text = parts[1].strip() if len(parts) > 1 else ''  # 后半作为正文
    tool_calls = []              # 收集工具调用
    for i, m in enumerate(re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)):  # 逐个匹配 <tool_call>..</tool_call>
        try:
            call = json.loads(m.strip())  # 解析其中的 JSON（含 name/arguments）
            tool_calls.append({"id": f"call_{int(time.time())}_{i}", "type": "function", "function": {"name": call.get("name", ""), "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False)}})  # 转成 OpenAI 工具调用结构
        except Exception:        # 解析失败则跳过该段
            pass
    if tool_calls:               # 若解析出工具调用，则把这些标签从正文里删掉
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    return text.strip(), reasoning_content, tool_calls or None  # 返回（正文, 思考, 工具调用或None）


def generate_stream_response(messages, temperature, top_p, max_tokens, tools=None, open_thinking=False):  # 流式生成的核心生成器
    try:
        new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools or None, open_thinking=open_thinking)[-max_tokens:]  # 套用 chat 模板（含工具/思考），并按 max_tokens 截断末尾
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)  # 编码并搬到设备

        queue = Queue()          # 创建输出队列
        streamer = CustomStreamer(tokenizer, queue)  # 创建流式器（写入该队列）

        def _generate():         # 在子线程里执行真正的生成
            model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                attention_mask=inputs.attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer  # 生成的 token 会通过 streamer 进入队列
            )

        Thread(target=_generate).start()  # 启动生成线程（主线程随后从队列消费）

        full_text = ""           # 累积已生成的全部文本
        emitted = 0              # 已经吐给客户端的字符位置
        thinking_ended = not bool(open_thinking)  # 是否已结束思考阶段（未开启思考则视为一开始就结束）

        while True:              # 不断从队列取文本片段
            text = queue.get()   # 阻塞获取（生成线程产出或结束哨兵）
            if text is None:     # 收到结束哨兵
                break
            full_text += text    # 累积

            if not thinking_ended:  # 仍在思考阶段：把内容作为 reasoning_content 吐出
                pos = full_text.find('</think>')  # 找思考结束标记
                if pos >= 0:     # 思考结束
                    thinking_ended = True
                    new_r = full_text[emitted:pos]  # 截至 </think> 之前的新思考内容
                    if new_r:
                        yield json.dumps({"choices": [{"delta": {"reasoning_content": new_r}}]}, ensure_ascii=False)  # 作为 reasoning_content 增量吐出
                    emitted = pos + len('</think>')  # 跳过 </think>
                    after = full_text[emitted:].lstrip('\n')  # </think> 之后的正文（去掉前导换行）
                    emitted = len(full_text) - len(after)  # 修正已吐位置
                    if after:
                        yield json.dumps({"choices": [{"delta": {"content": after}}]}, ensure_ascii=False)  # 把正文部分作为 content 吐出
                        emitted = len(full_text)
                else:            # 思考尚未结束：继续把增量作为 reasoning_content 吐出
                    new_r = full_text[emitted:]
                    if new_r:
                        yield json.dumps({"choices": [{"delta": {"reasoning_content": new_r}}]}, ensure_ascii=False)
                        emitted = len(full_text)
            else:                # 已进入正文阶段：把增量作为 content 吐出
                new_c = full_text[emitted:]
                if new_c:
                    yield json.dumps({"choices": [{"delta": {"content": new_c}}]}, ensure_ascii=False)
                    emitted = len(full_text)

        _, _, tool_calls = parse_response(full_text)  # 生成结束后，从完整文本里解析工具调用
        if tool_calls:           # 若有工具调用则单独吐一帧
            yield json.dumps({"choices": [{"delta": {"tool_calls": tool_calls}}]}, ensure_ascii=False)
        yield json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls" if tool_calls else "stop"}]}, ensure_ascii=False)  # 吐结束帧，标明结束原因

    except Exception as e:       # 出错则吐出错误信息
        yield json.dumps({"error": str(e)})


@app.post("/v1/chat/completions")  # 注册 OpenAI 兼容的对话补全接口
async def chat_completions(request: ChatRequest):  # 请求体自动按 ChatRequest 校验
    try:
        if request.stream:       # 流式分支
            return StreamingResponse(  # 返回 SSE 流
                (f"data: {chunk}\n\n" for chunk in generate_stream_response(  # 把每个 JSON 帧包成 SSE 的 data: 行
                    messages=request.messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                    tools=request.tools,
                    open_thinking=request.get_open_thinking()
                )),
                media_type="text/event-stream"  # SSE 媒体类型
            )
        else:                    # 非流式分支：一次性生成并返回完整 JSON
            new_prompt = tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=request.tools or None,
                open_thinking=request.get_open_thinking()
            )[-request.max_tokens:]  # 套模板并截断
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)  # 编码
            with torch.no_grad():  # 关闭梯度
                generated_ids = model.generate(  # 生成
                    inputs["input_ids"],
                    max_length=inputs["input_ids"].shape[1] + request.max_tokens,  # 总长度上限
                    do_sample=True,
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=request.top_p,
                    temperature=request.temperature
                )
                answer = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)  # 只解码新生成部分
            content, reasoning_content, tool_calls = parse_response(answer)  # 拆出正文/思考/工具调用
            message = {"role": "assistant", "content": content}  # 组装回复消息
            if reasoning_content:  # 有思考则附带
                message["reasoning_content"] = reasoning_content
            if tool_calls:       # 有工具调用则附带
                message["tool_calls"] = tool_calls
            return {             # 返回 OpenAI 兼容的完整响应体
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "minimind",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if tool_calls else "stop"  # 结束原因
                    }
                ]
            }
    except Exception as e:       # 兜底异常 → 返回 500
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":       # 作为脚本启动服务
    parser = argparse.ArgumentParser(description="Server for MiniMind")  # 参数解析器
    parser.add_argument('--load_from', default='../model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")  # 模型来源
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")  # 权重目录
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, dpo, reason, ppo_actor, grpo, spo）")  # 权重前缀
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")  # 可选 LoRA
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")  # 隐藏维度
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")  # 层数
    parser.add_argument('--max_seq_len', default=8192, type=int, help="最大序列长度")  # 最大序列长度
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")  # 是否 MoE
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")  # 长上下文外推
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")  # 设备
    args = parser.parse_args()   # 解析参数
    device = args.device         # 全局设备（init_model / 各接口都用到）
    model, tokenizer = init_model(args)  # 加载模型与分词器
    uvicorn.run(app, host="0.0.0.0", port=8998)  # 在 0.0.0.0:8998 启动服务
