---
name: sonetto-interaction-tools
description: SonettoHere 暂停等待/用户交互工具说明；目标项目已有 ask_user 系列工具。
---

# SonettoHere Interaction Tools

参考来源：`SonettoHere-main/tools/interaction/TOOL.md`。

SuperAgent 当前等价工具：

- `ask_user_qa`
- `ask_user_single_choice`
- `ask_user_multi_choice`
- `run_python` 的执行前确认

这些工具通过 WebSocket 的 `interaction_request` 暂停 Agent，等待用户选择或输入后继续。
