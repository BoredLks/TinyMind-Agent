---
name: sonetto-task-tools
description: 任务追踪工具：无状态任务清单追踪。已注册为外部工具插件。
---

# SonettoHere Task Tools

已迁移为 SuperAgent 外部工具插件（`tools/task_tracker_tools.py`）。纯本地操作，无外部依赖。

## 可用工具

### task_tracker
- **功能**：无状态任务清单追踪，返回统计摘要
- **参数**：`todos` (array, 必填) — 全量任务清单，每项含 `content`、`status`(pending/in_progress/completed)、`activeForm`(可选)
- **说明**：LLM 每次调用传入完整列表，工具返回统计（总数/各状态计数/当前任务），不维护内部状态
- **无需审批**