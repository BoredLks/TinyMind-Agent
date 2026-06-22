"""运行配置：统一从环境变量 / .env 读取，集中管理 LLM 后端、workspace、生成参数等。

设计目标是“零配置可跑、可被环境变量覆盖”。三种 LLM 后端：

- ``local`` ：进程内直接加载 MiniMind 权重（需要 torch + transformers，即仓库本身的依赖）。
- ``server``：把 MiniMind 当成 OpenAI 兼容服务（即先跑 ``scripts/serve_openai_api.py``），
              再用 langchain 的 ChatOpenAI 连接它——和 MokioAgent 的接入方式完全一致。
- ``mock``  ：确定性假模型，用于在没有 torch / 没有服务时跑通并测试整张图。
"""

from __future__ import annotations  # 启用延迟注解求值（支持 str | None 等写法）

import os                            # 读取环境变量
from dataclasses import dataclass, field  # 用 dataclass 定义配置；field 设默认工厂
from pathlib import Path             # 路径处理

try:  # python-dotenv 是可选依赖；缺失时也不影响纯环境变量方式
    from dotenv import load_dotenv   # 能从 .env 文件加载变量
except Exception:  # pragma: no cover - 依赖缺失时的降级
    def load_dotenv(*_args, **_kwargs):  # type: ignore  # 没装 dotenv 就用一个空操作替身
        return False


# 项目根目录 = 本文件所在包（minimind_agent）的上一级，即 minimind-master-agent/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 合法的 LLM 后端模式
VALID_LLM_MODES = ("local", "server", "mock")


def _env_int(name: str, default: int) -> int:  # 读取一个正整数环境变量
    """读取一个正整数环境变量，非法 / 非正回退默认值。"""
    try:
        value = int(os.getenv(name, str(default)))  # 取环境变量并转 int
    except (TypeError, ValueError):  # 非数字
        return default
    return value if value > 0 else default  # 必须为正，否则用默认


def _env_float(name: str, default: float) -> float:  # 读取一个浮点环境变量
    try:
        return float(os.getenv(name, str(default)))  # 取环境变量并转 float
    except (TypeError, ValueError):  # 非数字回退默认
        return default


@dataclass
class AgentConfig:                   # 整个 Agent 运行期的配置中枢
    """整个 Agent 运行期的配置中枢。"""

    # ---- LLM 后端 ----
    llm_mode: str = "local"                  # local / server / mock
    # local 模式：MiniMind 权重相关（与 eval_llm.py 默认值保持一致）
    weight: str = "full_sft"                 # 权重前缀
    hidden_size: int = 768                   # 隐藏层维度
    num_hidden_layers: int = 8               # 层数
    use_moe: bool = False                    # 是否 MoE
    device: str | None = None                # None=自动（cuda 优先）
    # server 模式：OpenAI 兼容服务地址（默认指向 serve_openai_api.py）
    base_url: str = "http://127.0.0.1:8998/v1"
    api_key: str = "minimind"                # 本地服务不校验，占位即可
    model_name: str = "minimind"             # 请求里的 model 字段
    # ---- 生成参数 ----
    temperature: float = 0.6                 # 偏低一点，让规划/代码更稳定
    top_p: float = 0.9                       # 核采样阈值
    max_new_tokens: int = 1024               # 最大生成长度
    # ---- workspace 沙箱 ----
    workspace: Path = field(default_factory=lambda: PROJECT_ROOT / "agent_workspace")  # 默认沙箱目录
    seed_samples: bool = True                # 是否在 workspace 内生成演示用样例文件
    # ---- 执行/重试 ----
    max_replans: int = 1                     # 失败后最多重规划（重试）几轮
    allow_exec: bool = False                 # code_agent 是否真正执行生成的代码（默认仅语法校验）
    exec_timeout: int = 10                   # 允许执行时的超时（秒）

    def backend_label(self) -> str:  # 人类可读的“真实来源”后端名（用于在输出里标注产物来源）
        """人类可读的“真实来源”后端名，用于在输出里标注产物到底谁生成的。"""
        return {
            "local": "MiniMind(本地权重)",
            "server": "MiniMind(OpenAI服务)",
            "mock": "MOCK假模型⚠(非真实模型)",
        }.get(self.llm_mode, self.llm_mode)  # 未知模式则原样返回

    @classmethod
    def from_env(cls, **overrides) -> "AgentConfig":  # 从 .env/环境变量构建配置，再叠加显式覆盖
        """从 .env / 环境变量构建配置，再叠加显式 overrides（CLI 参数优先级最高）。"""
        load_dotenv()                # 先加载 .env，使下面的 os.getenv 能读到
        mode = os.getenv("MINIMIND_LLM_MODE", "local").strip().lower()  # 读后端模式
        if mode not in VALID_LLM_MODES:  # 非法模式回退 local
            mode = "local"
        cfg = cls(                   # 用环境变量逐项构造配置
            llm_mode=mode,
            weight=os.getenv("MINIMIND_WEIGHT", "full_sft"),
            hidden_size=_env_int("MINIMIND_HIDDEN_SIZE", 768),
            num_hidden_layers=_env_int("MINIMIND_NUM_LAYERS", 8),
            use_moe=os.getenv("MINIMIND_USE_MOE", "0").strip() in {"1", "true", "True"},  # 字符串转布尔
            device=os.getenv("MINIMIND_DEVICE") or None,  # 空串视为 None（自动）
            base_url=os.getenv("MINIMIND_BASE_URL", "http://127.0.0.1:8998/v1"),
            api_key=os.getenv("MINIMIND_API_KEY", "minimind"),
            model_name=os.getenv("MINIMIND_MODEL", "minimind"),
            temperature=_env_float("MINIMIND_TEMPERATURE", 0.6),
            top_p=_env_float("MINIMIND_TOP_P", 0.9),
            max_new_tokens=_env_int("MINIMIND_MAX_NEW_TOKENS", 1024),
            workspace=Path(os.getenv("MINIMIND_WORKSPACE", str(PROJECT_ROOT / "agent_workspace"))),
            seed_samples=os.getenv("MINIMIND_SEED_SAMPLES", "1").strip() not in {"0", "false", "False"},  # 默认开
            max_replans=_env_int("MINIMIND_MAX_REPLANS", 1),
            allow_exec=os.getenv("MINIMIND_ALLOW_EXEC", "0").strip() in {"1", "true", "True"},  # 默认关
            exec_timeout=_env_int("MINIMIND_EXEC_TIMEOUT", 10),
        )
        for key, value in overrides.items():  # 叠加 CLI/调用方的显式覆盖（仅当值非 None 且字段存在）
            if value is not None and hasattr(cfg, key):
                setattr(cfg, key, value)
        # workspace 统一成展开后的绝对路径
        cfg.workspace = Path(cfg.workspace).expanduser()
        if cfg.llm_mode not in VALID_LLM_MODES:  # 覆盖后再次校验模式合法
            cfg.llm_mode = "local"
        return cfg
