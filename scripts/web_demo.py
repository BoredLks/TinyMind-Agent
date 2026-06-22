import random                   # 随机数与随机种子
import re                        # 正则：解析/格式化 <think> 与 <tool_call>
import json                      # 工具参数/结果的 JSON 编解码
import os                        # 扫描模型目录
from threading import Thread     # 用子线程跑生成，配合流式器边生成边显示

import torch                     # 推理
import numpy as np               # 设随机种子用
import streamlit as st           # 网页 UI 框架（本文件是一个 streamlit 应用）
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer  # 模型、分词器、可迭代流式器

st.set_page_config(page_title="MiniMind", initial_sidebar_state="collapsed")  # 设置网页标题与默认收起侧边栏

# 注入自定义 CSS：把按钮改成小圆形、微调页面边距等（下面整段是 CSS 字符串，原样保留）
st.markdown("""
    <style>
        /* 添加操作按钮样式 */
        .stButton button {
            border-radius: 50% !important;  /* 改为圆形 */
            width: 32px !important;         /* 固定宽度 */
            height: 32px !important;        /* 固定高度 */
            padding: 0 !important;          /* 移除内边距 */
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #666 !important;         /* 更柔和的颜色 */
            margin: 5px 10px 5px 0 !important;  /* 调整按钮间距 */
        }
        .stButton button:hover {
            border-color: #999 !important;
            color: #333 !important;
            background-color: #f5f5f5 !important;
        }
        .stMainBlockContainer > div:first-child {
            margin-top: -50px !important;
        }
        .stApp > div:last-child {
            margin-bottom: -35px !important;
        }
        
        /* 重置按钮基础样式 */
        .stButton > button {
            all: unset !important;  /* 重置所有默认样式 */
            box-sizing: border-box !important;
            border-radius: 50% !important;
            width: 18px !important;
            height: 18px !important;
            min-width: 18px !important;
            min-height: 18px !important;
            max-width: 18px !important;
            max-height: 18px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #888 !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            margin: 0 2px !important;  /* 调整这里的 margin 值 */
        }

    </style>
""", unsafe_allow_html=True)     # unsafe_allow_html：允许渲染上面的原始 HTML/CSS

device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备

# 多语言文本
LANG_TEXTS = {                   # 界面文案的中英双语字典
    'zh': {
        'settings': '模型设定调整',
        'history_rounds': '历史对话轮次',
        'max_length': '最大生成长度',
        'temperature': '温度',
        'thinking': '思考',
        'tools': '工具',
        'language': '语言',
        'send': '给 MiniMind 发送消息',
        'disclaimer': 'AI 生成内容可能存在错误，请仔细核实',
        'think_tip': '自适应思考，目前多轮对话或Tool Call共存时思考不稳定',
        'tool_select': '工具选择（最多4个）',
    },
    'en': {
        'settings': 'Model Settings',
        'history_rounds': 'History Rounds',
        'max_length': 'Max Length',
        'temperature': 'Temperature',
        'thinking': 'Thinking',
        'tools': 'Tools',
        'language': 'Language',
        'send': 'Send a message to MiniMind',
        'disclaimer': 'AI-generated content may be inaccurate, please verify',
        'think_tip': 'Adaptive thinking; may be unstable with multi-turn or Tool Call',
        'tool_select': 'Tool Selection (max 4)',
    }
}

def get_text(key):               # 按当前语言取文案；找不到则回退中文、再回退 key 本身
    lang = st.session_state.get('lang', 'en')  # 当前语言（默认英文）
    return LANG_TEXTS.get(lang, {}).get(key, LANG_TEXTS['zh'].get(key, key))

# 工具定义
TOOLS = [                        # 提供给模型的工具清单（与 eval_toolcall 类似，描述更简短）
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式"}}, "required": ["expression"]}}},  # 数学
    {"type": "function", "function": {"name": "get_current_time", "description": "获取当前时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},  # 时间
    {"type": "function", "function": {"name": "random_number", "description": "生成随机数", "parameters": {"type": "object", "properties": {"min": {"type": "integer"}, "max": {"type": "integer"}}, "required": ["min", "max"]}}},  # 随机数
    {"type": "function", "function": {"name": "text_length", "description": "计算文本长度", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},  # 文本长度
    {"type": "function", "function": {"name": "unit_converter", "description": "单位转换", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},  # 单位换算
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取天气", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},  # 天气
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "获取汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},  # 汇率
    {"type": "function", "function": {"name": "translate_text", "description": "翻译文本", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_lang": {"type": "string"}}, "required": ["text", "target_lang"]}}},  # 翻译
]

TOOL_SHORT_NAMES = {             # 工具名 → 界面上显示的中文短名
    'calculate_math': '数学', 'get_current_time': '时间', 'random_number': '随机',
    'text_length': '字数', 'unit_converter': '单位', 'get_current_weather': '天气',
    'get_exchange_rate': '汇率', 'translate_text': '翻译'
}

def execute_tool(tool_name, args):  # 工具的（演示性）执行：返回假数据
    import datetime               # 局部导入，仅本函数用到
    try:
        if tool_name == 'calculate_math':  # 数学：直接 eval（仅演示）
            return {"result": eval(args.get('expression', '0'))}
        elif tool_name == 'get_current_time':  # 时间：返回当前时间字符串
            tz = args.get('timezone', 'Asia/Shanghai')
            return {"result": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        elif tool_name == 'random_number':  # 随机数
            return {"result": random.randint(args.get('min', 0), args.get('max', 100))}
        elif tool_name == 'text_length':  # 文本长度
            return {"result": len(args.get('text', ''))}
        elif tool_name == 'unit_converter':  # 单位换算（仅回显，不真算）
            return {"result": f"{args.get('value', 0)} {args.get('from_unit', '')} = ? {args.get('to_unit', '')}"}
        elif tool_name == 'get_current_weather':  # 天气（固定假数据）
            return {"result": f"{args.get('city', 'Unknown')}: 晴, 7~10°C"}
        elif tool_name == 'get_exchange_rate':  # 汇率（固定假数据）
            return {"result": f"1 {args.get('from_currency', 'USD')} = 7.2 {args.get('to_currency', 'CNY')}"}
        elif tool_name == 'translate_text':  # 翻译（固定假数据）
            return {"result": f"[翻译结果]: hello world"}
        return {"result": "Unknown tool"}  # 未知工具
    except Exception as e:        # 出错返回错误信息
        return {"error": str(e)}


def process_assistant_content(content, is_streaming=False):  # 把模型回复里的 <tool_call>/<think> 标签渲染成好看的 HTML 折叠块
    # 处理tool_call标签，格式化显示
    if '<tool_call>' in content:  # 存在工具调用标签
        def format_tool_call(match):  # 把单个 <tool_call> 替换成带样式的卡片
            try:
                tc = json.loads(match.group(1))  # 解析其中 JSON
                name = tc.get('name', 'unknown')  # 工具名
                args = tc.get('arguments', {})  # 参数
                return f'<div style="background: rgba(80, 110, 150, 0.20); border: 1px solid rgba(140, 170, 210, 0.30); padding: 10px 12px; border-radius: 12px; margin: 6px 0;"><div style="font-size:12px;opacity:.75;display:block;margin:0 0 6px 0;line-height:1;">ToolCalling</div><div><b>{name}</b>: {json.dumps(args, ensure_ascii=False)}</div></div>'  # 渲染成卡片
            except:
                return match.group(0)  # 解析失败则原样保留
        content = re.sub(r'<tool_call>(.*?)</tool_call>', format_tool_call, content, flags=re.DOTALL)  # 替换所有工具调用标签

    # 流式生成且开启思考时，一开始就放到折叠里
    if is_streaming and st.session_state.get('enable_thinking', False) and '</think>' not in content and '<think>' not in content:  # 开了思考、流式、且还没出现 think 标签
        m = re.search(r'(\n\n(?:我是|您好|你好)[^\n]*)', content)  # 尝试用「正式回答的起手语」判断思考与正文的分界
        if m and m.start(1) > 5:  # 找到分界且前面有足够内容
            i = m.start(1)
            think_part = content[:i]  # 分界前 → 思考
            answer_part = content[i:]  # 分界后 → 正文
            return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">已思考</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">{think_part.strip()}</div></details>{answer_part}'  # 思考折叠 + 正文
        elif len(content) > 5:  # 还没出现分界，但已有内容 → 整体当作“思考中”
            return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">思考中...</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto; display: flex; flex-direction: column-reverse;"><div style="margin-bottom: auto;">{content.strip().replace(chr(10), "<br>")}</div></div></details>'

    if '<think>' in content and '</think>' in content:  # 思考有完整成对标签
        def format_think(match):  # 把 <think>..</think> 替换成“已思考”折叠块
            think_content = match.group(2)
            if think_content.replace('\n', '').strip():  # 不是全换行
                return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">已思考</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">{think_content.strip()}</div></details>'
            return ''             # 思考为空则丢弃
        content = re.sub(r'(<think>)(.*?)(</think>)', format_think, content, flags=re.DOTALL)

    if '<think>' in content and '</think>' not in content:  # 只有开始标签（流式进行中）
        def format_think_in_progress(match):  # 渲染成“思考中...”折叠块
            tc = match.group(1)
            return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">思考中...</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto; display: flex; flex-direction: column-reverse;"><div style="margin-bottom: auto;">{tc.strip().replace(chr(10), "<br>")}</div></div></details>'
        content = re.sub(r'<think>(.*?)$', format_think_in_progress, content, flags=re.DOTALL)

    if '<think>' not in content and '</think>' in content:  # 只有结束标签（开始标签被模板吃掉的情形）
        def format_think_no_start(match):  # 把开头到 </think> 当作思考
            think_content = match.group(1)
            if think_content.replace('\n', '').strip():
                return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">已思考</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">{think_content.strip()}</div></details>'
            return ''
        content = re.sub(r'(.*?)</think>', format_think_no_start, content, flags=re.DOTALL)

    return content                # 返回渲染后的 HTML 内容


@st.cache_resource               # streamlit 缓存：同一 model_path 只加载一次模型，避免每次交互都重载
def load_model_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True   # 允许加载仓库自带的自定义建模代码
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model = model.half().eval().to(device)  # 半精度、评估模式、搬到设备
    return model, tokenizer


def clear_chat_messages():       # 清空对话（删除会话状态里的消息）
    del st.session_state.messages
    del st.session_state.chat_messages


def init_chat_messages():        # 渲染已有历史消息；若无则初始化为空
    if "messages" in st.session_state:  # 已有历史
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "assistant":  # 助手消息：渲染（含 think/tool 折叠）
                st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
            else:                # 用户消息：渲染成右侧气泡
                st.markdown(
                    f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; background-color: #3d4450; border-radius: 22px; color: white;">{message["content"]}</div></div>',
                    unsafe_allow_html=True)

    else:                        # 无历史则初始化
        st.session_state.messages = []
        st.session_state.chat_messages = []

    return st.session_state.messages

def regenerate_answer(index):    # 重新生成：弹出最后一轮，重跑页面
    st.session_state.messages.pop()
    st.session_state.chat_messages.pop()
    st.rerun()


# 动态扫描模型目录
script_dir = os.path.dirname(os.path.abspath(__file__))  # 本脚本所在目录
MODEL_PATHS = {}                 # 可选模型字典：目录名 → [路径, 显示名]
for d in sorted(os.listdir(script_dir), reverse=True):  # 倒序遍历同级目录
    full_path = os.path.join(script_dir, d)
    if os.path.isdir(full_path) and not d.startswith('.') and not d.startswith('_'):  # 跳过隐藏/下划线目录
        if any(f.endswith(('.bin', '.safetensors', '.pt')) or os.path.exists(os.path.join(full_path, 'model.safetensors.index.json')) for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))):  # 目录里含权重文件才算一个模型
            MODEL_PATHS[d] = [d, d]
if not MODEL_PATHS:              # 没扫到任何模型
    MODEL_PATHS = {"No models found": ["", "No models"]}

# 模型选择
selected_model = st.sidebar.selectbox('Model', list(MODEL_PATHS.keys()), index=0)  # 侧边栏下拉选模型
model_path = MODEL_PATHS[selected_model][0]  # 选中模型的路径
slogan = f"我是 {MODEL_PATHS[selected_model][1]}，有什么可以帮你的？" if st.session_state.get('lang', 'en') == 'zh' else f"I am {MODEL_PATHS[selected_model][1]}, how can I help you?"  # 顶部标语（按语言）

st.sidebar.markdown('<hr style="margin: 12px 0 16px 0;">', unsafe_allow_html=True)  # 侧边栏分隔线

# 语言选择
lang_options = {'中文': 'zh', 'English': 'en'}  # 语言标签 → 代码
current_lang = st.session_state.get('lang', 'en')  # 当前语言
lang_index = 0 if current_lang == 'zh' else 1  # 单选默认项
lang_label = st.sidebar.radio('Language / 语言', list(lang_options.keys()), index=lang_index, horizontal=True)  # 横向单选
if lang_options[lang_label] != current_lang:  # 切换了语言
    st.session_state.lang = lang_options[lang_label]  # 更新
    st.rerun()                   # 重跑以应用

st.sidebar.markdown('<hr style="margin: 12px 0 16px 0;">', unsafe_allow_html=True)  # 分隔线

# 参数设置
st.session_state.history_chat_num = st.sidebar.slider(get_text('history_rounds'), 0, 8, 0, step=2)  # 历史轮次（0~8，步长 2）
st.session_state.max_new_tokens = st.sidebar.slider(get_text('max_length'), 256, 8192, 8192, step=1)  # 最大生成长度
st.session_state.temperature = st.sidebar.slider(get_text('temperature'), 0.6, 1.2, 0.90, step=0.01)  # 温度

st.sidebar.markdown('<hr style="margin: 12px 0 16px 0;">', unsafe_allow_html=True)  # 分隔线

# 功能开关
st.session_state.enable_thinking = st.sidebar.checkbox(get_text('thinking'), value=False, help=get_text('think_tip'))  # 思考开关
st.session_state.selected_tools = []  # 本次选中的工具（最多 4 个）
with st.sidebar.expander(get_text('tools')):  # 工具选择折叠区
    st.caption(get_text('tool_select'))
    selected_count = sum(1 for tool in TOOLS if st.session_state.get(f"tool_{tool['function']['name']}", False))  # 已勾选数量
    for tool in TOOLS:           # 逐个工具渲染勾选框
        name = tool['function']['name']
        short_name = TOOL_SHORT_NAMES.get(name, name)  # 显示短名
        checked = st.checkbox(short_name, key=f"tool_{name}", disabled=(selected_count >= 4 and not st.session_state.get(f"tool_{name}", False)))  # 已选满 4 个则禁用其余
        if checked and len(st.session_state.selected_tools) < 4:  # 勾选则加入（上限 4）
            st.session_state.selected_tools.append(name)

image_url = "https://www.modelscope.cn/api/v1/studio/gongjy/MiniMind/repo?Revision=master&FilePath=images%2Flogo2.png&View=true"  # 顶部 logo 图片地址

st.markdown(                     # 渲染顶部标题区（logo + 标语 + 免责声明）
    f'<div style="display: flex; flex-direction: column; align-items: center; text-align: center; margin: 0; padding: 0;">'
    '<div style="font-style: italic; font-weight: 900; margin: 0; padding-top: 4px; display: flex; align-items: center; justify-content: center; flex-wrap: wrap; width: 100%;">'
    f'<img src="{image_url}" style="width: 40px; height: 40px; "> '
    f'<span style="font-size: 26px; margin-left: 10px;">{slogan}</span>'
    '</div>'
    f'<span style="color: #bbb; font-style: italic; margin-top: 6px; margin-bottom: 10px;">{get_text("disclaimer")}</span>'
    '</div>',
    unsafe_allow_html=True
)


def setup_seed(seed):            # 固定随机种子，保证生成可复现
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # cudnn 走确定性算法
    torch.backends.cudnn.benchmark = False     # 关闭自动寻优（保证确定性）


def main():                      # 页面主逻辑：渲染历史 → 接收输入 → 流式生成（含多轮工具调用）
    model, tokenizer = load_model_tokenizer(model_path)  # 加载（带缓存）模型

    if "messages" not in st.session_state:  # 首次进入初始化会话状态
        st.session_state.messages = []       # 用于页面展示的消息
        st.session_state.chat_messages = []  # 真正喂给模型的消息（含 system/tool 等）

    messages = st.session_state.messages  # 展示用消息列表

    for i, message in enumerate(messages):  # 渲染历史消息
        if message["role"] == "assistant":
            st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; background-color: #3d4450; border-radius: 22px; color: white;">{message["content"]}</div></div>',
                unsafe_allow_html=True)

    prompt = st.chat_input(key="input", placeholder=get_text('send'))  # 底部输入框

    if hasattr(st.session_state, 'regenerate') and st.session_state.regenerate:  # 处理“重新生成”标记
        prompt = st.session_state.last_user_message  # 用上一条用户消息当作输入
        regenerate_index = st.session_state.regenerate_index
        delattr(st.session_state, 'regenerate')       # 清掉这些标记
        delattr(st.session_state, 'last_user_message')
        delattr(st.session_state, 'regenerate_index')

    if prompt:                   # 有新输入时
        st.markdown(             # 先把用户消息渲染成右侧气泡
            f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; background-color: #3d4450; border-radius: 22px; color: white;">{prompt}</div></div>',
            unsafe_allow_html=True)
        messages.append({"role": "user", "content": prompt[-st.session_state.max_new_tokens:]})  # 加入展示消息（按长度截断）
        st.session_state.chat_messages.append({"role": "user", "content": prompt[-st.session_state.max_new_tokens:]})  # 加入模型消息

        placeholder = st.empty()  # 占位容器，用于流式刷新助手回复

        random_seed = random.randint(0, 2 ** 32 - 1)  # 随机种子
        setup_seed(random_seed)   # 固定种子

        tools = [t for t in TOOLS if t['function']['name'] in st.session_state.get('selected_tools', [])] or None  # 本次启用的工具（没选则 None）
        sys_prompt = [] if tools else [{"role": "system", "content": "你是MiniMind，一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题。"}]  # 无工具时给个通用系统提示
        st.session_state.chat_messages = sys_prompt + st.session_state.chat_messages[-(st.session_state.history_chat_num + 1):]  # 拼系统提示 + 最近 N 轮历史
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}  # chat 模板参数
        if st.session_state.get('enable_thinking', False):  # 开启思考
            template_kwargs["open_thinking"] = True
        if tools:                # 有工具
            template_kwargs["tools"] = tools
        new_prompt = tokenizer.apply_chat_template(st.session_state.chat_messages, **template_kwargs)  # 套模板得到输入文本

        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)  # 编码

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)  # 可迭代流式器（可在主线程 for 它）
        generation_kwargs = {     # 生成参数
            "input_ids": inputs.input_ids,
            "max_length": inputs.input_ids.shape[1] + st.session_state.max_new_tokens,
            "num_return_sequences": 1,
            "do_sample": True,
            "attention_mask": inputs.attention_mask,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "temperature": st.session_state.temperature,
            "top_p": 0.85,
            "streamer": streamer,
        }

        Thread(target=model.generate, kwargs=generation_kwargs).start()  # 子线程跑生成，主线程消费流式输出

        answer = ""               # 累积本轮回复
        for new_text in streamer:  # 逐片接收
            answer += new_text
            placeholder.markdown(process_assistant_content(answer, is_streaming=True), unsafe_allow_html=True)  # 实时刷新显示

        full_answer = answer      # 完整回复（含后续工具轮）
        for _ in range(16):       # 最多 16 轮工具调用循环
            tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', answer, re.DOTALL)  # 解析本轮工具调用
            if not tool_calls:    # 没有则结束
                break
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})  # 把含工具调用的助手消息写回
            tool_results = []     # 收集工具结果卡片 HTML
            for tc_str in tool_calls:  # 逐个执行
                try:
                    tc = json.loads(tc_str.strip())  # 解析工具调用 JSON
                    result = execute_tool(tc.get('name', ''), tc.get('arguments', {}))  # 执行
                    st.session_state.chat_messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})  # 把结果回灌给模型
                    tool_results.append(f'<div style="background: rgba(90, 130, 110, 0.20); border: 1px solid rgba(150, 200, 170, 0.30); padding: 10px 12px; border-radius: 12px; margin: 6px 0;"><div style="font-size:12px;opacity:.75;display:block;margin:0 0 6px 0;line-height:1;">ToolCalled</div><div><b>{tc.get("name", "")}</b>: {json.dumps(result, ensure_ascii=False)}</div></div>')  # 结果卡片
                except:
                    pass          # 解析失败跳过
            full_answer += "\n" + "\n".join(tool_results) + "\n"  # 把结果卡片拼进完整回复
            placeholder.markdown(process_assistant_content(full_answer, is_streaming=True), unsafe_allow_html=True)  # 刷新显示
            new_prompt = tokenizer.apply_chat_template(st.session_state.chat_messages, **template_kwargs)  # 带上工具结果重新套模板
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)  # 重新编码
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)  # 新流式器
            generation_kwargs["input_ids"] = inputs.input_ids  # 更新生成参数
            generation_kwargs["attention_mask"] = inputs.attention_mask
            generation_kwargs["max_length"] = inputs.input_ids.shape[1] + st.session_state.max_new_tokens
            generation_kwargs["streamer"] = streamer
            Thread(target=model.generate, kwargs=generation_kwargs).start()  # 再次子线程生成
            answer = ""           # 累积这一工具轮之后的新回复
            for new_text in streamer:
                answer += new_text
                placeholder.markdown(process_assistant_content(full_answer + answer, is_streaming=True), unsafe_allow_html=True)  # 拼上之前的内容一起刷新
            full_answer += answer  # 累加
        answer = full_answer      # 最终回复 = 完整内容

        messages.append({"role": "assistant", "content": answer})  # 写入展示消息
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})  # 写入模型消息


if __name__ == "__main__":       # 作为 streamlit 脚本运行
    main()
