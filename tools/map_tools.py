"""SonettoHere map tools — migrated to SuperAgent plugin.

AMap (高德地图) tools (require AMAP_API_KEY):
- geocode: 地理编码
- nearby: 附近搜索
- transit: 公交路线
- cycling: 骑行路线
- fuzzy_addr: 模糊地址解析
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


def _get_amap_key() -> str:
    key = os.getenv("AMAP_API_KEY", "")
    if not key:
        try:
            from app.core import secrets
            key = secrets.get_provider_api_key("amap") or ""
        except Exception:
            pass
    return key


class GeocodeTool(Tool):
    spec = ToolSpec(
        name="geocode",
        description="地理编码：将地址转换为经纬度坐标。需要设置环境变量 AMAP_API_KEY。",
        parameters={
            "type": "object",
            "properties": {
                "address": {"type": "string", "description": "要查询的地址"},
                "city": {"type": "string", "description": "指定城市（可选）"},
            },
            "required": ["address"],
        },
        doc="# geocode\n\n地址转经纬度。需要 AMAP_API_KEY。",
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            import httpx
        except ImportError:
            return ToolResult(False, "", "httpx 未安装")
        key = _get_amap_key()
        if not key:
            return ToolResult(False, "", "需要 AMAP_API_KEY。请在设置中配置或设置环境变量。")
        address = str(args.get("address") or "")
        city = str(args.get("city") or "")
        if not address:
            return ToolResult(False, "", "address 不能为空")
        params = {"key": key, "address": address}
        if city:
            params["city"] = city
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get("https://restapi.amap.com/v3/geocode/geo", params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                return ToolResult(True, _ok(data), display={"kind": "json", "data": data})
        except Exception as exc:
            return ToolResult(False, "", f"地理编码失败: {exc}")


class NearbyTool(Tool):
    spec = ToolSpec(
        name="nearby_search",
        description="搜索指定坐标附近的 POI（兴趣点）。需要设置环境变量 AMAP_API_KEY。",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "经纬度，格式 'lng,lat'"},
                "keyword": {"type": "string", "description": "搜索关键词"},
                "radius": {"type": "integer", "description": "搜索半径（米），默认 1000"},
                "types": {"type": "string", "description": "POI 类型代码"},
            },
            "required": ["location"],
        },
        doc="# nearby_search\n\n搜索附近 POI。需要 AMAP_API_KEY。",
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            import httpx
        except ImportError:
            return ToolResult(False, "", "httpx 未安装")
        key = _get_amap_key()
        if not key:
            return ToolResult(False, "", "需要 AMAP_API_KEY。")
        location = str(args.get("location") or "")
        if not location:
            return ToolResult(False, "", "location 不能为空")
        params = {"key": key, "location": location, "radius": str(args.get("radius") or 1000)}
        if args.get("keyword"):
            params["keywords"] = args["keyword"]
        if args.get("types"):
            params["types"] = args["types"]
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get("https://restapi.amap.com/v3/place/around", params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                return ToolResult(True, _ok(data), display={"kind": "json", "data": data})
        except Exception as exc:
            return ToolResult(False, "", f"附近搜索失败: {exc}")


class TransitTool(Tool):
    spec = ToolSpec(
        name="transit_route",
        description="公交路线规划。需要设置环境变量 AMAP_API_KEY。",
        parameters={
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "起点经纬度 'lng,lat'"},
                "destination": {"type": "string", "description": "终点经纬度 'lng,lat'"},
                "city": {"type": "string", "description": "城市代码"},
            },
            "required": ["origin", "destination", "city"],
        },
        doc="# transit_route\n\n公交路线规划。需要 AMAP_API_KEY。",
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            import httpx
        except ImportError:
            return ToolResult(False, "", "httpx 未安装")
        key = _get_amap_key()
        if not key:
            return ToolResult(False, "", "需要 AMAP_API_KEY。")
        origin = str(args.get("origin") or "")
        destination = str(args.get("destination") or "")
        city = str(args.get("city") or "")
        if not origin or not destination or not city:
            return ToolResult(False, "", "origin, destination, city 不能为空")
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://restapi.amap.com/v3/direction/transit/integrated",
                    params={"key": key, "origin": origin, "destination": destination, "city": city},
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                return ToolResult(True, _ok(data), display={"kind": "json", "data": data})
        except Exception as exc:
            return ToolResult(False, "", f"公交路线查询失败: {exc}")


class CyclingTool(Tool):
    spec = ToolSpec(
        name="cycling_route",
        description="骑行路线规划。需要设置环境变量 AMAP_API_KEY。",
        parameters={
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "起点经纬度 'lng,lat'"},
                "destination": {"type": "string", "description": "终点经纬度 'lng,lat'"},
            },
            "required": ["origin", "destination"],
        },
        doc="# cycling_route\n\n骑行路线规划。需要 AMAP_API_KEY。",
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            import httpx
        except ImportError:
            return ToolResult(False, "", "httpx 未安装")
        key = _get_amap_key()
        if not key:
            return ToolResult(False, "", "需要 AMAP_API_KEY。")
        origin = str(args.get("origin") or "")
        destination = str(args.get("destination") or "")
        if not origin or not destination:
            return ToolResult(False, "", "origin 和 destination 不能为空")
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://restapi.amap.com/v4/direction/bicycling",
                    params={"key": key, "origin": origin, "destination": destination},
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                return ToolResult(True, _ok(data), display={"kind": "json", "data": data})
        except Exception as exc:
            return ToolResult(False, "", f"骑行路线查询失败: {exc}")


class FuzzyAddrTool(Tool):
    spec = ToolSpec(
        name="fuzzy_addr",
        description=(
            "模糊地址搜索。输入不完整或模糊的地址信息，返回匹配的地址列表和坐标。"
            "使用高德地图 API。需要 AMAP_API_KEY。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "keywords": {"type": "string", "description": "搜索关键词（模糊地址）"},
                "city": {"type": "string", "description": "限定城市（可选）"},
                "citylimit": {"type": "boolean", "description": "是否限制在 city 内搜索，默认 false"},
            },
            "required": ["keywords"],
        },
        doc="# fuzzy_addr\n\n模糊地址搜索。输入不完整地址也能匹配。需要 AMAP_API_KEY。",
    )

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            import httpx
        except ImportError:
            return ToolResult(False, "", "httpx 未安装")
        keywords = str(args.get("keywords") or "").strip()
        city = str(args.get("city") or "").strip()
        citylimit = bool(args.get("citylimit", False))
        if not keywords:
            return ToolResult(False, "", "keywords 不能为空")
        key = _get_amap_key()
        if not key:
            return ToolResult(False, "", "需要 AMAP_API_KEY。")
        try:
            params = {"key": key, "keywords": keywords, "offset": "10", "extensions": "base"}
            if city:
                params["city"] = city
            if citylimit:
                params["citylimit"] = "true"
            async with httpx.AsyncClient() as client:
                resp = await client.get("https://restapi.amap.com/v3/place/text", params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
            if data.get("status") != "1":
                return ToolResult(False, "", f"高德 API 错误: {data.get('info', 'unknown')}")
            pois = data.get("pois", [])
            results = []
            for poi in pois:
                loc = poi.get("location", "")
                parts = loc.split(",") if loc else []
                results.append({
                    "name": poi.get("name", ""),
                    "address": poi.get("address", ""),
                    "type": poi.get("type", ""),
                    "lng": float(parts[0]) if len(parts) == 2 else None,
                    "lat": float(parts[1]) if len(parts) == 2 else None,
                    "city": poi.get("cityname", ""),
                    "district": poi.get("adname", ""),
                })
            return ToolResult(True, _ok({"total": len(results), "results": results}),
                              display={"kind": "json", "data": {"total": len(results), "results": results}})
        except httpx.TimeoutException:
            return ToolResult(False, "", "高德 API 请求超时")
        except Exception as exc:
            return ToolResult(False, "", f"模糊地址搜索失败: {exc}")


def register(registry):
    registry.register(GeocodeTool())
    registry.register(NearbyTool())
    registry.register(TransitTool())
    registry.register(CyclingTool())
    registry.register(FuzzyAddrTool())
