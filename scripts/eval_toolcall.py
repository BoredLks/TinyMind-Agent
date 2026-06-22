import os                       # 路径处理
import sys                       # 修改模块搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 把项目根加入搜索路径，便于 `from model...`
import re                        # 正则：从文本里解析 <tool_call>
import json                      # JSON 编解码（工具参数与结果）
import time                      # 计时
import random                    # 随机数（随机种子、random_number 工具）
import argparse                  # 命令行参数
import warnings                  # 屏蔽告警
import torch                     # 本地推理
from datetime import datetime    # get_current_time 工具用
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer  # 分词器、通用模型、流式器
from openai import OpenAI        # API 后端：以 OpenAI 协议访问本地服务
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # MiniMind 配置与模型
from trainer.trainer_utils import setup_seed, get_model_params  # 设种子、打印参数量
warnings.filterwarnings('ignore')  # 忽略告警

TOOLS = [                        # 提供给模型的工具清单（OpenAI function calling 格式）
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式的结果，支持加减乘除、幂运算、开方等", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式，如123+456、2**10、sqrt(144)"}}, "required": ["expression"]}}},  # 数学计算
    {"type": "function", "function": {"name": "get_current_time", "description": "获取当前日期和时间，支持指定时区", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "description": "时区名称，如Asia/Shanghai、America/New_York", "default": "Asia/Shanghai"}}, "required": []}}},  # 当前时间
    {"type": "function", "function": {"name": "random_number", "description": "生成指定范围内的随机数", "parameters": {"type": "object", "properties": {"min": {"type": "integer", "description": "最小值", "default": 0}, "max": {"type": "integer", "description": "最大值", "default": 100}}, "required": []}}},  # 随机数
    {"type": "function", "function": {"name": "text_length", "description": "计算文本的字符数和单词数", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "要统计的文本"}}, "required": ["text"]}}},  # 文本长度
    {"type": "function", "function": {"name": "unit_converter", "description": "进行单位换算，支持长度、重量、温度等", "parameters": {"type": "object", "properties": {"value": {"type": "number", "description": "要转换的数值"}, "from_unit": {"type": "string", "description": "源单位，如km、miles、kg、pounds、celsius、fahrenheit"}, "to_unit": {"type": "string", "description": "目标单位"}}, "required": ["value", "from_unit", "to_unit"]}}},  # 单位换算
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取指定城市的当前天气信息，包括温度、湿度和天气状况", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "城市名称，如北京、上海、New York"}, "unit": {"type": "string", "description": "温度单位，celsius或fahrenheit", "enum": ["celsius", "fahrenheit"], "default": "celsius"}}, "required": ["location"]}}},  # 天气
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询两种货币之间的实时汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string", "description": "源货币代码，如USD、CNY、EUR"}, "to_currency": {"type": "string", "description": "目标货币代码，如USD、CNY、EUR"}}, "required": ["from_currency", "to_currency"]}}},  # 汇率
    {"type": "function", "function": {"name": "translate_text", "description": "将文本翻译成目标语言", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "要翻译的文本"}, "target_language": {"type": "string", "description": "目标语言，如english、chinese、japanese、french"}}, "required": ["text", "target_language"]}}},  # 翻译
]

MOCK_RESULTS = {                 # 各工具的「模拟实现」（演示用，返回假数据）；key 为工具名，value 为接收 args 的 lambda
    "calculate_math": lambda args: {"result": str(eval(str(args.get("expression", "0")).replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-").replace("²", "**2").replace("³", "**3").replace("（", "(").replace("）", ")")))},  # 把各种符号规整后用 eval 求值（仅演示，勿用于不可信输入）
    "get_current_time": lambda args: {"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "timezone": args.get("timezone", "Asia/Shanghai")},  # 返回当前时间
    "random_number": lambda args: {"result": random.randint(int(args.get("min", 0)), int(args.get("max", 100)))},  # 返回区间随机整数
    "text_length": lambda args: {"characters": len(args.get("text", "")), "words": len(args.get("text", "").split())},  # 返回字符数与单词数
    "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * 0.621371, 2), "from": f"{args.get('value', 0)} {args.get('from_unit', '')}", "to": args.get("to_unit", "")},  # 演示性换算（固定按 km→miles 系数）
    "get_current_weather": lambda args: {"city": args.get("location"), "temperature": "22°C", "humidity": "65%", "condition": "晴"},  # 返回固定天气
    "get_exchange_rate": lambda args: {"from": args.get("from_currency", ""), "to": args.get("to_currency", ""), "rate": 7.15},  # 返回固定汇率
    "translate_text": lambda args: {"translated": "hello world"},  # 返回固定翻译
}

TOOL_MAP = {t["function"]["name"]: t for t in TOOLS}  # 工具名 → 工具定义，便于按名取用

def get_tools(names):            # 按名字列表取出对应的工具定义子集
    return [TOOL_MAP[n] for n in names]

TEST_CASES = [                   # 自动测试用例：每条含一个 prompt 和它可见的工具名列表
    {"prompt": "帮我算一下 256 乘以 37 等于多少", "tools": ["calculate_math", "get_current_time"]},
    {"prompt": "现在几点了？", "tools": ["get_current_time", "random_number"]},
    {"prompt": "帮我把100公里换算成英里", "tools": ["unit_converter", "calculate_math"]},
    {"prompt": "帮我生成一个1到1000的随机数，然后计算它的平方", "tools": ["random_number", "calculate_math", "text_length"]},
    {"prompt": "北京今天天气怎么样？", "tools": ["get_current_weather", "get_current_time"]},
    {"prompt": "查一下美元兑人民币汇率", "tools": ["get_exchange_rate", "get_current_time"]},
    {"prompt": "把'你好世界'翻译成英文", "tools": ["translate_text", "text_length"]},
    {"prompt": "What is the weather in Tokyo? Also convert 30 celsius to fahrenheit.", "tools": ["get_current_weather", "unit_converter", "get_current_time"]},
]


def init_model(args):            # 加载本地 MiniMind 模型（API 后端时不调用此函数）
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)  # 分词器
    if 'model' in args.load_from:  # 原生 .pth 方式
        model = MiniMindForCausalLM(MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)))  # 构造模型
        moe_suffix = '_moe' if args.use_moe else ''  # MoE 后缀
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'  # 权重路径
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)  # 加载权重
    else:                              # transformers 格式
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)  # 打印参数量
    return model.half().eval().to(args.device), tokenizer  # 返回模型与分词器


def parse_tool_calls(text):      # 本地后端：从生成文本里抓出所有 <tool_call> 的 JSON（{name, arguments}）
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)  # 抓所有成对标签内的内容
    calls = []
    for m in matches:            # 逐个解析
        try:
            calls.append(json.loads(m.strip()))  # 解析 JSON
        except Exception:        # 解析失败跳过
            pass
    return calls


def parse_tool_call_from_text(content):  # API 后端：当服务端没结构化返回 tool_calls 时，从正文里兜底解析
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'  # 匹配标签内的 {..}
    matches = re.findall(pattern, content, re.DOTALL)
    if not matches:              # 没有则返回 None
        return None
    tool_calls = []
    for i, match in enumerate(matches):  # 逐个转成 OpenAI 风格的 tool_call 结构
        try:
            data = json.loads(match)
            tool_calls.append({
                "id": f"call_{i}",
                "function": {"name": data.get("name", ""), "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False)}
            })
        except Exception:
            pass
    return tool_calls if tool_calls else None


def execute_tool(call, arguments=None):  # 执行某个工具调用，返回（模拟的）结果字典
    name = call.get("name", "") if isinstance(call, dict) else call  # call 可能是 dict（本地）或纯名字（API）
    try:
        raw_args = call.get("arguments", {}) if isinstance(call, dict) else arguments  # 取参数
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args  # 参数可能是 JSON 字符串，转成 dict
    except Exception:
        args = {}                # 解析失败用空参数
    fn = MOCK_RESULTS.get(name)  # 找到对应的模拟实现
    if not fn:                   # 未知工具
        return {"error": f"未知工具: {name}"}
    try:
        return fn(args)          # 执行并返回结果
    except Exception as e:       # 执行报错
        return {"error": f"工具执行失败: {str(e)[:80]}"}


def generate(model, tokenizer, messages, tools, args):  # 本地后端：一次生成（含工具提示），返回文本
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)  # 流式打印
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=False)  # 套 chat 模板（带工具、不思考）
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(args.device)  # 编码
    st = time.time()             # 计时起点
    print('🧠: ', end='')         # 回答前缀
    generated_ids = model.generate(  # 生成
        inputs["input_ids"], attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        top_p=args.top_p, temperature=args.temperature
    )
    response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)  # 只取新生成部分
    gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])  # 生成 token 数
    print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s') if args.show_speed else print()  # 打印速度
    return response


def chat_api(client, messages, tools, args, stream=True):  # API 后端：调用 OpenAI 兼容服务，返回（正文, 工具调用）
    response = client.chat.completions.create(  # 发起请求
        model=args.api_model, messages=messages, tools=tools,
        stream=stream, temperature=args.temperature,
        max_tokens=8192, top_p=args.top_p
    )
    if not stream:               # 非流式：直接取完整结果
        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = choice.message.tool_calls  # 结构化工具调用
        if not tool_calls:       # 没有则从正文兜底解析
            tool_calls = parse_tool_call_from_text(content)
        print(f'🧠: {content}')
        return content, tool_calls
    print('🧠: ', end='', flush=True)  # 流式前缀
    content, tool_calls = "", None  # 累积正文与工具调用
    for chunk in response:       # 逐块接收
        delta = chunk.choices[0].delta
        if delta.content:        # 正文增量
            print(delta.content, end="", flush=True)
            content += delta.content
        if delta.tool_calls:     # 工具调用增量（可能分多块到达，需要按 index 拼接）
            if tool_calls is None:
                tool_calls = []
            for tc_chunk in delta.tool_calls:
                idx = tc_chunk.index if tc_chunk.index is not None else len(tool_calls)  # 该调用所在槽位
                while len(tool_calls) <= idx:  # 槽位不够则补齐
                    tool_calls.append({
                        "id": "",
                        "function": {"name": "", "arguments": ""}
                    })
                if tc_chunk.id:  # 拼接 id
                    tool_calls[idx]["id"] += tc_chunk.id
                if tc_chunk.function:  # 拼接函数名与参数（参数是逐字符流式到达的）
                    if tc_chunk.function.name:
                        tool_calls[idx]["function"]["name"] += tc_chunk.function.name
                    if tc_chunk.function.arguments:
                        tool_calls[idx]["function"]["arguments"] += tc_chunk.function.arguments
    print()
    if not tool_calls:           # 流式结束后仍无结构化工具调用 → 从正文兜底解析
        tool_calls = parse_tool_call_from_text(content)
    return content, tool_calls


def run_case(prompt, tools, args, model=None, tokenizer=None, client=None):  # 跑一条用例：循环「生成→若有工具调用则执行并回灌→再生成」直到无工具调用
    messages = [{"role": "user", "content": prompt}]  # 初始只有用户问题
    while True:                  # 多轮工具调用循环
        if args.backend == 'local':  # 本地后端
            content = generate(model, tokenizer, messages, tools, args)  # 生成文本
            tool_calls = parse_tool_calls(content)  # 解析工具调用
        else:                    # API 后端
            content, tool_calls = chat_api(client, messages, tools, args, stream=bool(args.stream))  # 生成并解析
        if not tool_calls:       # 没有工具调用 → 本轮对话结束
            break
        tool_calls = [{          # 把不同后端的工具调用统一成 {id, name, arguments} 形式
            "id": tc.id if hasattr(tc, 'id') else tc.get("id", ""),
            "name": tc.function.name if hasattr(tc, 'function') else tc["function"]["name"],
            "arguments": tc.function.arguments if hasattr(tc, 'function') else tc["function"]["arguments"]
        } for tc in tool_calls] if args.backend == 'api' else tool_calls  # 仅 API 后端需要这步归一化
        messages.append({"role": "assistant", "content": content} if args.backend == 'local' else {"role": "assistant", "content": content, "tool_calls": [{"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}} for tc in tool_calls]})  # 把模型这轮（含工具调用）写回对话
        for tc in tool_calls:    # 逐个执行被请求的工具
            name = tc["name"]
            arguments = tc["arguments"]
            print(f'📞 [Tool Calling]: {name} | args={arguments}')  # 打印调用
            result = execute_tool(tc if args.backend == 'local' else name, arguments)  # 执行（本地传整个 dict，API 传名字+参数）
            print(f'✅ [Tool Called]: {json.dumps(result, ensure_ascii=False)}')  # 打印结果
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)} if args.backend == 'local' else {"role": "tool", "content": json.dumps(result, ensure_ascii=False), "tool_call_id": tc["id"]})  # 把工具结果回灌给模型（API 需带 tool_call_id）


def main():                      # 命令行主流程
    parser = argparse.ArgumentParser(description="MiniMind ToolCall评测")  # 参数解析器
    parser.add_argument('--backend', default='local', choices=['local', 'api'], type=str, help="推理后端（local=本地模型，api=OpenAI兼容接口）")  # 后端选择
    parser.add_argument('--load_from', default='../model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")  # 模型来源
    parser.add_argument('--save_dir', default='../out', type=str, help="模型权重目录")  # 权重目录
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")  # 权重前缀
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")  # 隐藏维度
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")  # 层数
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")  # 是否 MoE
    parser.add_argument('--max_new_tokens', default=512, type=int, help="最大生成长度")  # 生成上限
    parser.add_argument('--temperature', default=0.9, type=float, help="生成温度，控制随机性（0-1，越大越随机）")  # 温度
    parser.add_argument('--top_p', default=0.9, type=float, help="nucleus采样阈值（0-1）")  # 核采样
    parser.add_argument('--show_speed', default=0, type=int, help="显示decode速度（tokens/s）")  # 是否显示速度
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")  # 设备
    parser.add_argument('--api_base_url', default="http://localhost:11434/v1", type=str, help="OpenAI兼容接口的base_url")  # API 地址
    parser.add_argument('--api_key', default='sk-123', type=str, help="OpenAI兼容接口的api_key")  # API 密钥（占位）
    parser.add_argument('--api_model', default='jingyaogong/minimind-3:latest', type=str, help="API请求时使用的模型名称")  # API 模型名
    parser.add_argument('--stream', default=1, type=int, help="API模式下是否流式输出（0=否，1=是）")  # 是否流式
    args = parser.parse_args()   # 解析

    model = tokenizer = client = None  # 预置变量
    if args.backend == 'local': model, tokenizer = init_model(args)  # 本地后端：加载模型
    else: client = OpenAI(api_key=args.api_key, base_url=args.api_base_url)  # API 后端：建客户端

    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))  # 模式：0=跑内置用例，1=手动输入

    cases = [{"prompt": case["prompt"], "tools": get_tools(case["tools"]), "tool_names": case["tools"]} for case in TEST_CASES] if input_mode == 0 else iter(lambda: {"prompt": input('💬: '), "tools": TOOLS, "tool_names": [t["function"]["name"] for t in TOOLS]}, {"prompt": "", "tools": TOOLS, "tool_names": []})  # 构造用例迭代器（手动模式下每次读输入，空串结束）
    for case in cases:           # 逐条处理
        if not case["prompt"]: break  # 空输入则退出
        setup_seed(random.randint(0, 31415926))  # 每条用例设随机种子
        if input_mode == 0:      # 自动模式打印可用工具与问题
            print(f'📦 可用工具: {case["tool_names"]}\n')
            print(f'💬: {case["prompt"]}')
        run_case(case["prompt"], case["tools"], args, model=model, tokenizer=tokenizer, client=client)  # 执行该用例
        print('\n' + '-' * 50 + '\n')  # 分隔线


if __name__ == "__main__":       # 作为脚本运行
    main()
