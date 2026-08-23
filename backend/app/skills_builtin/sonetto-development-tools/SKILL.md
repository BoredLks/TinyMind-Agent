---
name: sonetto-development-tools
description: 代码开发工具：代码质量分析、语法检查、调试助手、单元测试执行。已注册为外部工具插件。
---

# SonettoHere Development Tools

已迁移为 SuperAgent 外部工具插件（`tools/development_tools.py`）。纯本地操作，无外部依赖。

## 可用工具

### code_quality
- **功能**：分析 Python 代码质量（复杂度/可维护性/重复代码）
- **参数**：`code` / `file_path`、`analysis_type`（complexity/maintainability/duplication/all）
- **无需审批**

### syntax_check
- **功能**：检查 Python 代码语法错误
- **参数**：`code` / `file_path`
- **无需审批**

### debug_helper
- **功能**：分析错误信息并提供调试建议
- **参数**：`error_message`（必填）、`code_context`（可选）
- **无需审批**

### unit_test_runner
- **功能**：执行 Python 单元测试文件，返回通过/失败/错误统计报告
- **参数**：`test_file`（必填）、`test_class`（可选）、`test_method`（可选）
- **无需审批**