---
name: sonetto-todo-tools
description: 本地待办事项管理：添加、列出、完成、取消完成、更新、删除、查询、列出项目。已注册为外部工具插件。
---

# SonettoHere Todo Tools

已迁移为 SuperAgent 外部工具插件（`tools/todo_tools.py`）。使用本地 SQLite 存储，无需外部 API。

## 可用工具

### todo_add — 添加待办
- **参数**：`content`（必填）、`project`、`priority`(1-4)、`due_date`(YYYY-MM-DD)

### todo_list — 列出待办
- **参数**：`project`、`completed`(bool)、`limit`

### todo_complete — 标记完成
- **参数**：`id`（必填）

### todo_uncomplete — 取消完成
- **参数**：`id`（必填）

### todo_update — 更新待办
- **参数**：`id`（必填）、`content`、`project`、`priority`、`due_date`

### todo_delete — 删除待办（需审批）
- **参数**：`id`（必填）

### todo_query — 按关键词查询待办
- **参数**：`keyword`（必填）、`limit`

### todo_list_projects — 列出所有项目名称
- **参数**：无