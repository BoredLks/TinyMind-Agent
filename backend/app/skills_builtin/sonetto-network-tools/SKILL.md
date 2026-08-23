---
name: sonetto-network-tools
description: 网络工具：天气查询、假日查询、图片理解。已注册为外部工具插件。
---

# SonettoHere Network Tools

已迁移为 SuperAgent 外部工具插件（`tools/network_tools.py`）。

## 可用工具

### weather
- **功能**：获取指定城市的天气信息（实时/预报/逐小时/生活指数）
- **参数**：`city` / `adcode`、`extended`、`forecast`、`hourly`、`indices`、`lang`
- **依赖**：需要 `UAPIS_API_KEY` 环境变量
- **无需审批**

### holiday
- **功能**：查询中国法定假日信息
- **参数**：`year`（必填）、`month`（可选）
- **依赖**：需要 `UAPIS_API_KEY` 环境变量
- **无需审批**

### analyze_image
- **功能**：使用多模态模型理解图片内容。支持本地文件和网络图片。
- **参数**：`image_source`（必填, 'local:path' 或 'url:https://...'）、`prompt`（默认"请描述这张图片"）
- **依赖**：需要配置支持视觉的 LLM provider
- **无需审批**