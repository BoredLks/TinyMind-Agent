"""SonettoHere network tools — migrated to SuperAgent plugin.

External API tools (require API keys or network access):
- weather: 天气查询 (需要 uapis API key)
- holiday: 假日查询 (需要 uapis API key)
- image_understand: 图片理解 (需要 uapis API key)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from app.tools.base import Tool, ToolContext, ToolResult, ToolSpec


def _ok(data: dict) -> str:
    return json.dumps({"success": True, "data": data}, ensure_ascii=False)


def _err(message: str) -> str:
    return json.dumps({"success": False, "error": message}, ensure_ascii=False)


_UAPIS_BASE = "https://uapis.cn"


class WeatherTool(Tool):
    spec = ToolSpec(
        name="weather",
        description=(
            "获取指定城市的天气信息。支持实时天气、多天预报、逐小时预报、分钟级降水、生活指数。"
            "city 和 adcode 二选一即可。需要设置环境变量 UAPIS_API_KEY。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称，如'北京'、'Shanghai'"},
                "adcode": {"type": "string", "description": "行政区划代码，如'110000'"},
                "extended": {"type": "boolean", "description": "返回体感温度/能见度/气压/紫外线/AQI 等扩展信息"},
                "forecast": {"type": "boolean", "description": "返回最多7天预报"},
                "hourly": {"type": "boolean", "description": "返回24小时逐小时预报"},
                "indices": {"type": "boolean", "description": "返回生活指数"},
                "lang": {"type": "string", "description": "语言：zh/en，默认 zh"},
            },
        },
        doc="# weather\n\n获取城市天气信息。需要 UAPIS_API_KEY 环境变量或在 settings 中配置。",
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            import httpx
        except ImportError:
            return ToolResult(False, "", "httpx 未安装")

        api_key = os.getenv("UAPIS_API_KEY", "")
        if not api_key:
            try:
                from app.core import secrets
                api_key = secrets.get_provider_api_key("uapis") or ""
            except Exception:
                pass
        if not api_key:
            return ToolResult(False, "", "天气查询需要 UAPIS_API_KEY。请在设置中配置或设置环境变量。")

        city = str(args.get("city") or "")
        adcode = str(args.get("adcode") or "")
        if not city and not adcode:
            return ToolResult(False, "", "city 和 adcode 至少提供一个")

        params: dict = {}
        if city:
            params["city"] = city
        if adcode:
            params["adcode"] = adcode
        for key in ("extended", "forecast", "hourly", "indices"):
            if args.get(key):
                params[key] = "true"
        if args.get("lang"):
            params["lang"] = args["lang"]

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{_UAPIS_BASE}/api/v1/misc/weather",
                    params=params,
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                return ToolResult(True, _ok(data), display={"kind": "json", "data": data})
        except Exception as exc:
            return ToolResult(False, "", f"天气查询失败: {exc}")


class HolidayTool(Tool):
    spec = ToolSpec(
        name="holiday",
        description="查询中国法定假日信息。需要设置环境变量 UAPIS_API_KEY。",
        parameters={
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "年份，如 2024"},
                "month": {"type": "integer", "description": "月份（可选），1-12"},
            },
            "required": ["year"],
        },
        doc="# holiday\n\n查询中国法定假日。需要 UAPIS_API_KEY。",
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            import httpx
        except ImportError:
            return ToolResult(False, "", "httpx 未安装")

        api_key = os.getenv("UAPIS_API_KEY", "")
        if not api_key:
            try:
                from app.core import secrets
                api_key = secrets.get_provider_api_key("uapis") or ""
            except Exception:
                pass
        if not api_key:
            return ToolResult(False, "", "假日查询需要 UAPIS_API_KEY。")

        year = args.get("year")
        month = args.get("month")
        if not year:
            return ToolResult(False, "", "必须提供 year 参数")

        params: dict = {"year": str(year)}
        if month:
            params["month"] = str(month)

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{_UAPIS_BASE}/api/v1/misc/holiday",
                    params=params,
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                return ToolResult(True, _ok(data), display={"kind": "json", "data": data})
        except Exception as exc:
            return ToolResult(False, "", f"假日查询失败: {exc}")


class AnalyzeImageTool(Tool):
    spec = ToolSpec(
        name="analyze_image",
        description=(
            "使用多模态模型理解图片内容。支持本地文件（local:path）和网络图片（url:https://...）。"
            "可指定 prompt 提问，如 '这张图里有什么文字？'。"
            "需要配置支持视觉的模型（如 GLM-5V、GPT-4V 等）和对应的 API Key。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "image_source": {
                    "type": "string",
                    "description": "图片来源：'local:文件路径' 或 'url:https://...'，文件路径相对于工作区",
                },
                "prompt": {
                    "type": "string",
                    "description": "向模型提问的指令，默认 '请描述这张图片'",
                },
            },
            "required": ["image_source"],
        },
        doc=(
            "# analyze_image\n\n"
            "使用多模态视觉模型理解图片。\n"
            "- 支持本地文件：local:path\n"
            "- 支持网络图片：url:https://...\n"
            "需要配置支持视觉的 LLM provider 和 API Key。"
        ),
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        import base64
        import mimetypes
        from pathlib import Path

        image_source = str(args.get("image_source") or "").strip()
        prompt = str(args.get("prompt") or "请描述这张图片").strip()

        if not image_source:
            return ToolResult(False, "", "image_source 不能为空")

        try:
            if image_source.startswith("local:"):
                file_path = image_source[6:]
                from app.tools.workspace import resolve_in_workspace, PathEscapeError
                try:
                    p = resolve_in_workspace(ctx.workspace_root, file_path)
                except PathEscapeError as exc:
                    return ToolResult(False, "", str(exc))
                if not p.exists() or not p.is_file():
                    return ToolResult(False, "", f"文件不存在: {file_path}")
                image_bytes = p.read_bytes()
                ext = p.suffix.lower()
                mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp"}
                mime = mime_map.get(ext, "image/png")
            elif image_source.startswith("url:"):
                url = image_source[4:]
                import httpx
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.get(url, headers={"User-Agent": "SuperAgent/1.0"})
                    resp.raise_for_status()
                    ct = resp.headers.get("Content-Type", "")
                    mime = ct.split(";")[0].strip() if "/" in ct else "image/png"
                    image_bytes = resp.content
            else:
                return ToolResult(False, "", "image_source 必须以 'local:' 或 'url:' 开头")

            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            data_url = f"data:{mime};base64,{image_b64}"

            # Use the active LLM provider for vision
            from app.core import config as app_config
            settings = app_config.load_settings(getattr(ctx, "db", None))
            providers = [p for p in app_config.load_provider_records(getattr(ctx, "db", None)) if p.enabled]
            active_id = settings.active_provider_id
            provider = next((p for p in providers if p.id == active_id), providers[0] if providers else None)
            if not provider:
                return ToolResult(False, "", "未配置 LLM provider")

            from app.core import config as _cfg
            api_key = _cfg._resolve_provider_api_key(provider.id)
            if not api_key:
                return ToolResult(False, "", f"Provider {provider.label} API Key 未配置")

            from openai import AsyncOpenAI
            client = AsyncOpenAI(base_url=provider.base_url, api_key=api_key)
            response = await client.chat.completions.create(
                model=provider.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                max_tokens=1024,
            )
            result_text = response.choices[0].message.content or ""
            result = {"response": result_text, "model": provider.model}
            return ToolResult(True, _ok(result), display={"kind": "json", "data": result})
        except Exception as exc:
            return ToolResult(False, "", f"图片理解失败: {exc}")


def register(registry):
    registry.register(WeatherTool())
    registry.register(HolidayTool())
    registry.register(AnalyzeImageTool())
