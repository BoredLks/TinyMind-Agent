"""SonettoHere bilibili tools — migrated to SuperAgent plugin.

- bilibili_download: 下载 B 站视频 (需要 cookie + yt-dlp)
- bilibili_set_cookie: 设置/保存 B 站 cookie
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from app.tools.base import Tool, ToolContext, ToolResult, ToolSpec


def _ok(data: dict) -> str:
    return json.dumps({"success": True, "data": data}, ensure_ascii=False)


def _err(message: str) -> str:
    return json.dumps({"success": False, "error": message}, ensure_ascii=False)


_COOKIE_FILE = os.path.join(os.path.expanduser("~"), "SuperAgent", "bilibili_cookie.txt")


class BilibiliSetCookieTool(Tool):
    spec = ToolSpec(
        name="bilibili_set_cookie",
        description=(
            "设置 B 站 (Bilibili) Cookie，用于下载需要登录的视频。"
            "Cookie 可从浏览器开发者工具 > Application > Cookies 中复制。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "cookie": {"type": "string", "description": "B 站完整 Cookie 字符串"},
            },
            "required": ["cookie"],
        },
        doc="# bilibili_set_cookie\n\n保存 B 站 Cookie 供 bilibili_download 使用。Cookie 约 30 天过期，需定期更新。",
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        cookie = str(args.get("cookie") or "").strip()
        if not cookie:
            return ToolResult(False, "", "cookie 不能为空")
        try:
            os.makedirs(os.path.dirname(_COOKIE_FILE), exist_ok=True)
            Path(_COOKIE_FILE).write_text(cookie, encoding="utf-8")
            result = {"saved": True, "file": _COOKIE_FILE}
            return ToolResult(True, _ok(result), display={"kind": "json", "data": result})
        except Exception as exc:
            return ToolResult(False, "", f"保存失败: {exc}")


class BilibiliDownloadTool(Tool):
    spec = ToolSpec(
        name="bilibili_download",
        description=(
            "下载 B 站 (Bilibili) 视频。支持 /video/BV* 和 /video/av* 格式。"
            "需要先用 bilibili_set_cookie 设置有效的 Cookie。"
            "需要安装 yt-dlp（pip install yt-dlp）。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "B 站视频链接，如 https://www.bilibili.com/video/BVxxx/"},
                "quality": {
                    "type": "string",
                    "enum": ["highest", "1080P", "720P", "480P", "360P"],
                    "description": "画质，默认 highest",
                },
                "output_dir": {"type": "string", "description": "输出目录（项目内相对路径），默认 .superagent_downloads"},
            },
            "required": ["url"],
        },
        requires_approval=True,
        doc="# bilibili_download\n\n下载 B 站视频。需先设置 Cookie。需要 yt-dlp。",
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        url = str(args.get("url") or "").strip()
        quality = str(args.get("quality") or "highest")
        output_dir = str(args.get("output_dir") or ".superagent_downloads")

        if not url:
            return ToolResult(False, "", "url 不能为空")

        # Check cookie
        if not os.path.exists(_COOKIE_FILE):
            return ToolResult(False, "", "Cookie 未配置。请先使用 bilibili_set_cookie 设置 B 站 Cookie。")
        cookie = Path(_COOKIE_FILE).read_text(encoding="utf-8").strip()
        if not cookie:
            return ToolResult(False, "", "Cookie 文件为空。请使用 bilibili_set_cookie 重新设置。")

        # Check yt-dlp
        import shutil
        ytdlp = shutil.which("yt-dlp")
        if not ytdlp:
            return ToolResult(False, "", "yt-dlp 未安装。请运行: pip install yt-dlp")

        # Resolve output directory
        from app.tools.workspace import resolve_in_workspace, PathEscapeError
        try:
            out_p = resolve_in_workspace(ctx.workspace_root, output_dir)
        except PathEscapeError as exc:
            return ToolResult(False, "", str(exc))
        out_p.mkdir(parents=True, exist_ok=True)

        # Map quality
        quality_map = {"highest": "0", "1080P": "80", "720P": "64", "480P": "32", "360P": "16"}
        fmt_code = quality_map.get(quality, "0")

        try:
            import asyncio

            def _download():
                import subprocess
                cookie_path = _COOKIE_FILE
                cmd = [
                    ytdlp,
                    "--cookies", cookie_path,
                    "-f", f"bv*[height<=1080]+ba/b" if quality == "highest" else f"bv*[format_id={fmt_code}]+ba/b",
                    "--merge-output-format", "mp4",
                    "-o", str(out_p / "%(title)s.%(ext)s"),
                    "--no-playlist",
                    url,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                return result

            proc = await asyncio.to_thread(_download)
            if proc.returncode == 0:
                # Find downloaded file
                files = [f.name for f in out_p.iterdir() if f.is_file()]
                result = {"success": True, "output_dir": str(out_p), "files": files, "quality": quality}
                return ToolResult(True, _ok(result), display={"kind": "json", "data": result})
            else:
                return ToolResult(False, "", f"下载失败: {proc.stderr[:500]}")
        except subprocess.TimeoutExpired:
            return ToolResult(False, "", "下载超时（300秒）")
        except Exception as exc:
            return ToolResult(False, "", f"下载失败: {exc}")


def register(registry):
    registry.register(BilibiliSetCookieTool())
    registry.register(BilibiliDownloadTool())