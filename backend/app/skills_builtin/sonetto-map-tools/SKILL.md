---
name: sonetto-map-tools
description: 地图工具：地理编码、附近搜索、公交/骑行路线规划、模糊地址搜索。已注册为外部工具插件。
---

# SonettoHere Map Tools

已迁移为 SuperAgent 外部工具插件（`tools/map_tools.py`）。

## 可用工具

### geocode
- **功能**：地址转经纬度坐标
- **参数**：`address`（必填）、`city`（可选）
- **依赖**：需要 `AMAP_API_KEY`

### nearby_search
- **功能**：搜索指定坐标附近的 POI
- **参数**：`location`（必填, 'lng,lat'）、`keyword`、`radius`、`types`
- **依赖**：需要 `AMAP_API_KEY`

### transit_route
- **功能**：公交路线规划
- **参数**：`origin`、`destination`、`city`（均为必填）
- **依赖**：需要 `AMAP_API_KEY`

### cycling_route
- **功能**：骑行路线规划
- **参数**：`origin`、`destination`（均为必填）
- **依赖**：需要 `AMAP_API_KEY`

### fuzzy_addr
- **功能**：模糊地址搜索，输入不完整地址也能匹配
- **参数**：`keywords`（必填）、`city`（可选）、`citylimit`（可选）
- **依赖**：需要 `AMAP_API_KEY`