---
name: sonetto-files-tools
description: 文件处理工具：PDF 阅读、Word 文档阅读、文件精确编辑。已注册为外部工具插件。
---

# SonettoHere Files Tools

已迁移为 SuperAgent 外部工具插件。插件文件：`tools/files_tools.py`（PDF/DOC）、`tools/file_edit_tools.py`（精确编辑）。

## 可用工具

### pdf_reader
- **功能**：读取 PDF 文件（元数据/文本/搜索/目录）
- **参数**：`operation`、`file_path`、`start_page`/`end_page`、`query`
- **依赖**：`pip install PyPDF2`

### doc_reader
- **功能**：读取 Word 文档（元数据/文本/段落/表格）
- **参数**：`operation`、`file_path`、`start_paragraph`/`end_paragraph`、`query`
- **依赖**：`pip install python-docx`

### file_edit
- **功能**：文件精确编辑（old_string 精确替换、多笔编辑、行范围读取、正则搜索）
- **参数**：`operation`（edit/multi_edit/read/search）、`file_path`、`old_string`、`new_string`、`edits`、`offset`、`limit`、`pattern`
- **无需依赖**