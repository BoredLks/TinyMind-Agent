---
name: sonetto-bilibili-tools
description: Bilibili 工具：视频下载 + Cookie 管理。已注册为外部工具插件。
---

# SonettoHere Bilibili Tools

已迁移为 SuperAgent 外部工具插件（`tools/bilibili_tools.py`）。

## 可用工具

### bilibili_set_cookie
- **功能**：设置 B 站 Cookie，供下载工具使用
- **参数**：`cookie` (string, 必填) — 完整 Cookie 字符串
- **说明**：Cookie 可从浏览器开发者工具 > Application > Cookies 中复制，约 30 天过期
- **无需审批**

### bilibili_download
- **功能**：下载 B 站视频
- **参数**：
  - `url` (string, 必填) — 视频链接，支持 BV* 和 av* 格式
  - `quality` (string) — 画质：highest/1080P/720P/480P/360P
  - `output_dir` (string) — 输出目录
- **依赖**：需要先设置 Cookie，需要安装 yt-dlp (`pip install yt-dlp`)
- **需要审批**

## 使用流程
1. 首次使用：先调用 `bilibili_set_cookie` 设置 Cookie
2. 下载视频：调用 `bilibili_download` 传入视频 URL
3. Cookie 过期后需重新设置