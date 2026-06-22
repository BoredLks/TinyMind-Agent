from openai import OpenAI  # OpenAI 官方 SDK；用它以 OpenAI 兼容协议访问本地部署的 MiniMind 服务

client = OpenAI(           # 创建客户端，指向本地的 OpenAI 兼容服务（如 serve_openai_api.py 或 ollama）
    api_key="sk-123",      # 本地服务一般不校验密钥，占位即可
    base_url="http://localhost:11434/v1"  # 服务地址（此处为 ollama 默认端口 11434；用 serve_openai_api.py 时改成 8998）
)
stream = True              # 是否流式输出（True=边生成边打印）
conversation_history_origin = []  # 原始空对话（用作重置基准）
conversation_history = conversation_history_origin.copy()  # 实际使用的对话历史（拷贝一份，避免改动原始列表）
history_messages_num = 0  # 必须设置为偶数（Q+A），为0则不携带历史对话
while True:                # 循环：不断读取用户输入并请求模型
    query = input('[Q]: ')  # 读取一行用户问题
    conversation_history.append({"role": "user", "content": query})  # 把问题加入对话历史
    response = client.chat.completions.create(  # 调用 chat completions 接口
        model="minimind-local:latest",  # 模型名（需与服务端注册的名字一致）
        messages=conversation_history[-(history_messages_num or 1):],  # 取最近 N 条历史；为 0 时取最后 1 条（即只发当前问题）
        stream=stream,         # 是否流式
        temperature=0.8,       # 采样温度（越大越随机）
        max_tokens=2048,       # 最大生成长度
        top_p=0.8,             # 核采样阈值
        extra_body={"chat_template_kwargs": {"open_thinking": True}, "reasoning_effort": "medium"} # 思考开关
    )
    if not stream:             # 非流式：一次性拿到完整回复
        assistant_res = response.choices[0].message.content  # 取回复正文
        print('[A]: ', assistant_res)  # 打印
    else:                      # 流式：逐块接收并打印
        print('[A]: ', end='', flush=True)  # 先打印前缀，不换行
        assistant_res = ''     # 累积完整回复正文
        for chunk in response:  # 遍历流式返回的每个数据块
            delta = chunk.choices[0].delta  # 本块的增量内容
            r = getattr(delta, 'reasoning_content', None) or ""  # 思考内容（若服务端返回了 reasoning_content）
            c = delta.content or ""  # 正文内容
            if r:               # 有思考内容：用灰色打印（\033[90m..\033[0m 是 ANSI 灰色转义）
                print(f'\033[90m{r}\033[0m', end="", flush=True)
            if c:               # 有正文：正常打印
                print(c, end="", flush=True)
            assistant_res += c  # 只把正文累积进回复（思考过程不计入对话历史）

    conversation_history.append({"role": "assistant", "content": assistant_res})  # 把回复写回历史，支持多轮对话
    print('\n\n')              # 打印两个空行作分隔
