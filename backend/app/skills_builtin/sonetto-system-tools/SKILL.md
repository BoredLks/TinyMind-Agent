---
name: sonetto-system-tools
description: SonettoHere 系统工具说明；time_tool 已迁移，Python 执行由 run_python 承担。
---

# SonettoHere System Tools

参考来源：`SonettoHere-main/tools/system/TOOL.md`。

已迁移/对应：

- `time_tool`：当前日期时间。
- `run_python`：执行 Python 代码，执行前需要用户确认。

命令执行使用 SuperAgent 的 `run_command`，默认 PowerShell，并受项目目录限制。
