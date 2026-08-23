---
name: sonetto-memory-tools
description: SonettoHere 记忆工具说明；list/read/create/update/delete/merge 已迁为 SuperAgent 原生工具。
---

# SonettoHere Memory Tools

参考来源：`SonettoHere-main/tools/memory/TOOL.md`。

已迁移为 SuperAgent 原生工具：

- `list_memories`
- `read_memories`
- `create_memory`
- `update_memory`
- `delete_memory`
- `merge_memories`

这些工具操作 SuperAgent 当前 SQLite 长期记忆，与后台长期记忆整理 Worker 共用同一数据源。
