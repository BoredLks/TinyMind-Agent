"""workspace 沙箱：把所有文件操作严格限制在一个工作目录内（参考 MokioAgent 的边界护栏）。

三个专家 Agent 真正“动手”的能力都收口到这里：移动文件、写文本文件、Python 语法校验、
（可选）执行代码。所有路径都会被解析成 workspace 内的绝对路径并做越界校验，避免误伤
工作区之外的文件。
"""

from __future__ import annotations  # 启用延迟注解求值

import shutil                        # 移动文件（shutil.move）
import subprocess                    # 可选地在子进程里执行生成的代码
import sys                           # 取当前解释器路径（执行代码时用）
from dataclasses import dataclass    # 用 dataclass 定义 Workspace
from pathlib import Path             # 路径处理
from typing import Any               # 返回的结果字典值类型不定


# 读取文本时依次尝试的编码（兼容中文 Windows 的 GBK）
TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "gbk")


def _read_text_lossy(path: Path) -> str:  # 多编码容错读取，绝不抛错
    """多编码容错读取，最终用 replace 兜底，绝不抛错。"""
    for encoding in TEXT_ENCODINGS:  # 依次尝试每种编码
        try:
            return path.read_text(encoding=encoding)  # 某种编码成功就返回
        except UnicodeDecodeError:   # 解码失败则换下一种
            continue
    return path.read_text(encoding="utf-8", errors="replace")  # 全失败：用替换模式兜底（坏字节→�）


@dataclass
class Workspace:                     # 一次运行绑定的工作区；所有读写都不允许越过 root
    """一次运行绑定的工作区。所有读写都不允许越过 ``root``。"""

    root: Path                       # 工作区根目录

    def __post_init__(self) -> None:  # dataclass 初始化后自动调用
        self.root = Path(self.root).expanduser().resolve()  # 展开 ~ 并转绝对路径
        self.root.mkdir(parents=True, exist_ok=True)        # 确保目录存在

    # ------------------------------------------------------------------ #
    # 路径解析与安全校验
    # ------------------------------------------------------------------ #
    def resolve(self, path_str: str) -> Path:  # 把任意路径解析成 workspace 内的绝对路径，并校验不越界
        """把用户/模型给的路径解析成 workspace 内的绝对路径，并校验不越界。"""
        cleaned = str(path_str).strip().strip('"').strip("'").strip()  # 去首尾空白与引号
        cleaned = cleaned.replace("\\", "/")  # 反斜杠统一成正斜杠
        # 去掉模型常误加的 workspace/ 前缀
        for prefix in ("./workspace/", "workspace/", "./"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]  # 剥掉该前缀
        raw = Path(cleaned).expanduser()  # 展开 ~
        if not raw.is_absolute():    # 相对路径
            raw = self.root / raw    # 拼到 workspace 下
        resolved = raw.resolve()     # 解析成绝对路径（消除 .. 等）
        root = self.root.resolve()
        if resolved != root and root not in resolved.parents:  # 既不是 root 本身，也不在 root 之下 → 越界
            raise ValueError(f"路径越界，必须位于 workspace 内: {resolved}")
        return resolved              # 合法则返回

    def display(self, path: Path) -> str:  # 转成相对 workspace 的展示路径，便于阅读
        """转成相对 workspace 的展示路径，便于阅读。"""
        try:
            return str(Path(path).resolve().relative_to(self.root))  # 相对路径
        except ValueError:           # 不在 root 下则原样返回绝对路径
            return str(path)

    # ------------------------------------------------------------------ #
    # file_agent 用：移动文件
    # ------------------------------------------------------------------ #
    def move_file(self, source: str, destination: str, *, allow_overwrite: bool = False) -> dict[str, Any]:  # 移动文件
        """把 source 移动到 destination（都限制在 workspace 内）。

        - destination 若是已存在的目录、或以 / 结尾，则视为“移动进该目录、保留原文件名”。
        - 默认拒绝覆盖已存在的目标文件。
        返回结构化结果字典（ok / error / from / to ...）。
        """
        try:
            src = self.resolve(source)  # 解析源路径（含越界校验）
        except ValueError as exc:    # 越界
            return {"ok": False, "error": str(exc), "source": source}
        if not src.exists():         # 源不存在
            return {"ok": False, "error": f"源文件不存在: {self.display(src)}", "source": self.display(src)}

        dest_is_dir_hint = str(destination).strip().endswith(("/", "\\"))  # 目标以分隔符结尾 → 提示它是目录
        try:
            dst = self.resolve(destination)  # 解析目标路径
        except ValueError as exc:
            return {"ok": False, "error": str(exc), "destination": destination}

        # 目标是目录（已存在的目录 / 以分隔符结尾的提示）→ 移动进去，保留文件名
        if dst.is_dir() or dest_is_dir_hint:
            dst.mkdir(parents=True, exist_ok=True)  # 确保目录存在
            dst = dst / src.name     # 目标 = 目录/原文件名

        if dst.resolve() == src.resolve():  # 源与目标相同
            return {"ok": False, "error": "源和目标相同，无需移动", "from": self.display(src)}
        if dst.exists() and not allow_overwrite:  # 目标已存在且不允许覆盖
            return {
                "ok": False,
                "error": f"目标已存在，已阻止覆盖: {self.display(dst)}",
                "from": self.display(src),
                "to": self.display(dst),
            }

        dst.parent.mkdir(parents=True, exist_ok=True)  # 确保目标父目录存在
        shutil.move(str(src), str(dst))  # 真正移动（确定性操作，不经过模型）
        return {                     # 返回成功结果
            "ok": True,
            "action": "move",
            "from": self.display(src),
            "to": self.display(dst),
            "message": f"已移动 {self.display(src)} → {self.display(dst)}",
        }

    # ------------------------------------------------------------------ #
    # code_agent / story_agent 用：写文本文件
    # ------------------------------------------------------------------ #
    def write_text(self, path: str, content: str, *, allow_overwrite: bool = True) -> dict[str, Any]:  # 写文本文件
        """在 workspace 内创建/写入一个文本文件。"""
        try:
            target = self.resolve(path)  # 解析目标路径
        except ValueError as exc:
            return {"ok": False, "error": str(exc), "path": path}
        existed = target.exists()    # 记录原本是否存在（用于区分 create/update）
        if existed and not allow_overwrite:  # 已存在且不允许覆盖
            return {"ok": False, "error": f"文件已存在: {self.display(target)}", "path": self.display(target)}
        target.parent.mkdir(parents=True, exist_ok=True)  # 确保父目录存在
        target.write_text(content, encoding="utf-8")      # 以 UTF-8 写入
        return {                     # 返回结果（含字节数/行数等元信息）
            "ok": True,
            "action": "update" if existed else "create",
            "path": self.display(target),
            "abs_path": str(target),
            "bytes": len(content.encode("utf-8")),
            "lines": len(content.splitlines()),
        }

    def read_text(self, path: str) -> dict[str, Any]:  # 读取文本文件（供网页/校验用）
        try:
            target = self.resolve(path)  # 解析路径
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        if not target.is_file():     # 不是文件
            return {"ok": False, "error": f"文件不存在: {self.display(target)}"}
        return {"ok": True, "path": self.display(target), "content": _read_text_lossy(target)}  # 返回内容

    # ------------------------------------------------------------------ #
    # code_agent 用：语法校验 / 可选执行
    # ------------------------------------------------------------------ #
    def syntax_check(self, path: str) -> dict[str, Any]:  # Python 语法校验（只编译不执行）
        """语法校验（只编译、不执行、也不落 .pyc）。"""
        try:
            target = self.resolve(path)  # 解析路径
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        try:
            source = _read_text_lossy(target)  # 读出源码
            compile(source, str(target), "exec")  # 内置 compile：仅检查语法，不写 __pycache__
            return {"ok": True, "path": self.display(target), "message": "语法校验通过"}
        except SyntaxError as exc:   # 语法错误：报出行号与原因
            return {"ok": False, "path": self.display(target), "error": f"语法错误: 第{exc.lineno}行 {exc.msg}"}

    def run_python(self, path: str, *, timeout: int = 10) -> dict[str, Any]:  # 可选：子进程执行生成的代码
        """可选：在子进程里执行生成的 Python（默认关闭，需配置 allow_exec）。"""
        try:
            target = self.resolve(path)  # 解析路径
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        try:
            proc = subprocess.run(   # 用当前解释器在 workspace 目录里执行该脚本
                [sys.executable, str(target)],
                cwd=str(self.root),  # 工作目录固定为沙箱根
                capture_output=True,  # 捕获标准输出/错误
                text=True,
                timeout=timeout,     # 超时保护
            )
        except subprocess.TimeoutExpired:  # 超时
            return {"ok": False, "error": f"执行超时（>{timeout}s）"}
        return {                     # 返回退出码与截断后的输出
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": (proc.stdout or "")[:2000],
            "stderr": (proc.stderr or "")[:2000],
        }

    # ------------------------------------------------------------------ #
    # 演示辅助：生成几个样例文件，便于直接体验“移动文件”
    # ------------------------------------------------------------------ #
    def seed_samples(self) -> list[str]:  # 生成演示样例文件（已存在则跳过）
        """在 workspace/inbox 下生成几个样例文件（已存在则跳过）。返回创建的文件列表。"""
        created: list[str] = []      # 记录本次实际创建的文件
        samples = {                  # 样例文件：相对路径 → 内容
            "inbox/note.txt": "这是一个用于演示“移动文件”能力的样例文本文件。\n",
            "inbox/data.csv": "name,score\nminimind,99\n",
            "inbox/readme.md": "# 样例\n\n把我移动到别的目录试试。\n",
        }
        for rel, text in samples.items():
            target = self.root / rel  # 拼出绝对路径
            if not target.exists():  # 不存在才创建（避免覆盖用户已有文件）
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
                created.append(self.display(target))
        # 提供一个空的归档目录作为常见移动目标
        (self.root / "archive").mkdir(parents=True, exist_ok=True)
        return created

    def list_files(self) -> list[str]:  # 列出 workspace 内所有文件（相对路径）
        """列出 workspace 内的所有文件（相对路径），便于展示与校验。"""
        files = [self.display(p) for p in sorted(self.root.rglob("*")) if p.is_file()]  # 递归遍历取所有文件
        return files
