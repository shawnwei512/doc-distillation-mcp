# Doc Distillation MCP Server

A [Model Context Protocol](https://modelcontextprotocol.io) server for document distillation with multi-source support, producing dual output: HTML distillation articles and Obsidian notes.

## Features

- **Multi-source support**: Feishu documents, webpages, PDFs, video/podcast transcripts, and local files
- **Dual output format**: HTML distillation articles + Obsidian notes with frontmatter
- **Five-stage workflow**: Source extraction → Integrity safeguard → Image filtering → HTML generation → Obsidian generation
- **Three-layer image filtering**: Automatic rule filtering → Context prediction → Safety net
- **Key element detection**: Formulas, data, templates, checklists, frameworks, tables, warnings, quotes
- **Structure skeleton**: Heading-based document outline for integrity verification
- **Sync & Async modes**: Direct results for small documents, task polling for large ones
- **Structured output**: Pydantic-validated results with segments, images, and metadata

## Quick Start

### Install

```bash
pip install doc-distillation-mcp

# With dev tools (MCP Inspector, testing, linting)
pip install 'doc-distillation-mcp[dev]'
```

### Run

```bash
# Direct run
doc-distillation-mcp

# Or with uvx (no install needed)
uvx doc-distillation-mcp

# Debug with MCP Inspector
mcp dev doc_distillation_mcp.server:mcp
```

### Prerequisites (optional)

For PDF text extraction:

```bash
pip install pdfplumber
```

## MCP Client Configuration

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "doc-distillation": {
      "command": "uvx",
      "args": ["doc-distillation-mcp"]
    }
  }
}
```

### Cursor

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "doc-distillation": {
      "command": "uvx",
      "args": ["doc-distillation-mcp"]
    }
  }
}
```

### Trae

Add to Trae MCP settings:

```json
{
  "mcpServers": {
    "doc-distillation": {
      "command": "python3",
      "args": ["-m", "doc_distillation_mcp.server"]
    }
  }
}
```

### Claude Code

```bash
claude mcp add doc-distillation -- uvx doc-distillation-mcp
```

## Tools

### `distill_url`

Distill content from a URL into an HTML article + Obsidian note.

```python
# Webpage (sync mode - direct result)
distill_url(url="https://example.com/article")

# With Obsidian subdirectory
distill_url(
    url="https://example.com/deep-dive",
    obsidian_subdir="飞书蒸馏"
)

# Large document (async mode - returns task_id)
distill_url(
    url="https://example.com/long-report.pdf",
    async_mode=True
)
# Then poll:
get_distill_status(task_id="abc12345")
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | str | required | Document URL (Feishu, webpage, PDF, video) |
| `obsidian_subdir` | str? | null | Subdirectory under Obsidian vault |
| `async_mode` | bool | false | Return task_id for polling |

### `distill_file`

Distill content from a local file.

```python
# Text file (sync mode)
distill_file(file_path="/path/to/notes.txt")

# PDF file with Obsidian subdirectory
distill_file(
    file_path="/path/to/report.pdf",
    obsidian_subdir="PDF蒸馏"
)

# Large file (async mode)
distill_file(
    file_path="/path/to/large.pdf",
    async_mode=True
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | str | required | Path to local file |
| `obsidian_subdir` | str? | null | Subdirectory under Obsidian vault |
| `async_mode` | bool | false | Return task_id for polling |

### `get_distill_status`

Poll the status of an async distillation task.

```python
get_distill_status(task_id="abc12345")
# Returns: {status: "completed", progress: 1.0, result: {...}}
```

### `list_distillations`

List all completed distillations.

```python
list_distillations()
# Returns: [{task_id, title, source_type, method, segment_count, ...}]
```

## Five-Stage Workflow

```
URL / File Input
    │
    ├─ Stage 1: Source Detection & Content Extraction
    │   ├─ Feishu: Returns guidance (use lark-doc skill)
    │   ├─ Webpage: HTML parsing (text, headings, images)
    │   ├─ PDF: pdfplumber text extraction
    │   ├─ Video/Podcast: Returns guidance (use video-transcript-mcp)
    │   └─ Local file: Type-based extraction
    │
    ├─ Stage 2: Integrity Safeguard
    │   ├─ Structure skeleton (heading hierarchy)
    │   └─ Key element detection (8 categories)
    │
    ├─ Stage 2.5: Image Filtering (three-layer)
    │   ├─ Layer 1: Automatic rules (size, URL keywords, duplicates, alt keywords)
    │   ├─ Layer 2: Context prediction (nearby text indicates value)
    │   └─ Layer 3: Safety net (near key elements)
    │
    ├─ Stage 3: HTML Distillation Article Generation
    │   └─ Styled HTML with header, content sections, footer
    │
    └─ Stage 4: Obsidian Note Generation
        ├─ Frontmatter (title, source, author, date, tags)
        ├─ Summary callout
        ├─ Content sections
        ├─ Image distillation callouts
        └─ Key element statistics table
```

## Key Element Detection

The distiller detects and counts 8 types of key elements to ensure content completeness:

| Element | Description | Example Patterns |
|---------|-------------|-----------------|
| `formula` | Calculation formulas | `ROI =`, `= 销售额`, division |
| `data` | Numeric data | Percentages, amounts, multiples |
| `template` | Templates & scripts | Title formulas, word lists |
| `checklist` | Actionable lists | Numbered items, checkboxes |
| `framework` | Mental models | Matrices, quadrants, methodologies |
| `table` | Tabular data | Markdown tables, comparison |
| `warning` | Cautions & pitfalls | "Don't", "Avoid", "Pitfall" |
| `quote` | Notable quotes | Long quoted text, key phrases |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DISTILL_HTML_DIR` | `~/Documents/蒸馏文稿` | HTML output directory |
| `DISTILL_OBSIDIAN_DIR` | `~/Documents/obsidian` | Obsidian vault directory |

## Supported Sources

| Source | URL | Local File | Notes |
|--------|:---:|:---:|-------|
| Webpage | ✅ | ✅ | HTML parsing with image extraction |
| PDF | ✅ | ✅ | Requires `pdfplumber` |
| Feishu | ✅ | N/A | Returns guidance (use lark-doc skill) |
| YouTube | ✅ | N/A | Returns guidance (use video-transcript-mcp) |
| Bilibili | ✅ | N/A | Returns guidance (use video-transcript-mcp) |
| Douyin | ✅ | N/A | Returns guidance (use video-transcript-mcp) |
| Xiaohongshu | ✅ | N/A | Returns guidance (mini-program) |
| Text/Markdown | N/A | ✅ | Direct text extraction |
| Audio/Video | N/A | ✅ | Returns guidance (use video-transcript-mcp) |

## Community

Join our **AI Tool Monetization Circle** (AI 工具变现实战圈) on Knowledge Planet (知识星球):

- Weekly MCP tutorials and real-world case studies
- Deep-dive source code analysis of this project
- AI tool monetization strategies and playbooks
- 1-on-1 technical Q&A

> Scan the QR code below or search "AI 工具变现实战圈" on Knowledge Planet to join.

![Knowledge Planet QR Code](knowledge-planet-qr.jpg)

## License

MIT
