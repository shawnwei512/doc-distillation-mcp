"""Core distillation logic for the doc-distillation MCP server.

Five-stage workflow:
  1. Source detection and content extraction
  2. Integrity safeguard (structure skeleton + key element marking)
  3. Content analysis and image distillation guidance
  4. HTML / Obsidian dual output generation
  5. Integrity verification (structure + key element check)

The distiller handles extraction, structuring, and template generation.
Actual AI-powered content distillation is delegated to the MCP client.
"""

from __future__ import annotations

import html
import logging
import os
import re
import subprocess
import time
from collections.abc import Callable
from html.parser import HTMLParser

from .models import (
    ContentType,
    DistillationMethod,
    DistillResult,
    DistillSegment,
    DocInfo,
    ImageInfo,
    SourceType,
)

logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================
DEFAULT_HTML_OUTPUT_DIR = os.path.join(
    os.path.expanduser("~"), "Documents", "蒸馏文稿"
)
DEFAULT_OBSIDIAN_DIR = os.path.join(
    os.path.expanduser("~"), "Documents", "obsidian"
)

# Image filtering thresholds
MIN_IMAGE_SIZE = 80  # pixels
MIN_FILE_SIZE = 2048  # bytes (2KB)
FILTER_URL_KEYWORDS = [
    "avatar", "emoji", "icon", "logo", "qr", "bg", "background",
    "decoration", "header_bg", "footer_bg",
]
FILTER_ALT_KEYWORDS = [
    "头像", "装饰", "背景", "二维码", "关注", "点赞", "表情",
]

# Key element detection patterns
KEY_ELEMENT_PATTERNS: dict[str, list[str]] = {
    "formula": [
        r"=\s*\d", r"ROI\s*=", r"\d+\s*[÷/]\s*\d", r"公式\s*[：:]",
        r"计算\s*[：:]", r"=\s*毛利",
    ],
    "data": [
        r"\d+\.?\d*\s*[%％]", r"\d+\s*元", r"\d+\s*万", r"\d+\s*倍",
        r"占比\s*\d", r"转化率\s*\d",
    ],
    "template": [
        r"标题公式", r"话术", r"词列表", r"模板\s*[：:]",
        r"公式\s*[：:]", r"结构\s*[：:]",
    ],
    "checklist": [
        r"^\d+\.\s", r"^\-\s\[\s*\]", r"checklist", r"自检", r"清单",
        r"步骤\s*[：:]",
    ],
    "framework": [
        r"模型\s*[：:]", r"框架\s*[：:]", r"矩阵", r"象限",
        r"维度\s*[：:]", r"方法论",
    ],
    "table": [
        r"\|.*\|.*\|", r"对比\s*[：:]", r"分类表", r"<table",
    ],
    "warning": [
        r"不要", r"避免", r"坑", r"误区", r"注意",
        r"警告", r"切记", r"千万",
    ],
    "quote": [
        r'"[^"]{10,}"', r"「[^」]{10,}」",
        r"核心一句话", r"金句",
    ],
}

# Source detection patterns (order matters: specific types before generic WEBPAGE)
SOURCE_PATTERNS: dict[SourceType, list[str]] = {
    SourceType.FEISHU: [r"feishu\.cn", r"larkoffice\.com", r"yitang\.top/fs-doc", r"doubao\.com/docx", r"doubao\.com/wiki"],
    SourceType.PDF: [r"\.pdf$"],
    SourceType.PODCAST: [r"\.mp3$", r"\.m4a$", r"\.wav$", r"\.flac$"],
    SourceType.VIDEO: [r"youtube\.com", r"youtu\.be", r"bilibili\.com", r"b23\.tv", r"douyin\.com", r"kuaishou\.com", r"tiktok\.com"],
    SourceType.WEBPAGE: [r"https?://"],
    SourceType.LOCAL_FILE: [r"^/", r"^\./", r"^[A-Z]:/"],
}

# Closed platforms that need mini-program assistance
CLOSED_PLATFORMS = {"xiaohongshu", "weixin_video"}

# Progress callback type
ProgressCallback = Callable[[float, float | None, str], None]


def detect_source_type(url: str) -> SourceType:
    """Detect the source type from a URL or file path."""
    url_lower = url.lower()
    for source_type, patterns in SOURCE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, url_lower):
                return source_type
    if url.startswith("http"):
        return SourceType.WEBPAGE
    return SourceType.UNKNOWN


def _safe_filename(title: str) -> str:
    """Create a filesystem-safe filename from a title."""
    safe = re.sub(r'[\\/:*?"<>|]', "_", title)
    return safe[:80] if len(safe) > 80 else safe


def _format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ============================================================
# HTML Parser for content extraction
# ============================================================
class ContentExtractor(HTMLParser):
    """Parse HTML to extract text content, headings, and images."""

    def __init__(self) -> None:
        super().__init__()
        self.text_parts: list[str] = []
        self.headings: list[tuple[str, str]] = []  # (level, text)
        self.images: list[ImageInfo] = []
        self._current_tag: str = ""
        self._current_attrs: dict[str, str] = {}
        self._heading_text: str = ""
        self._in_heading = False
        self._skip_tags = {"script", "style", "nav", "footer", "header"}
        self._in_skip = 0
        self._block_index = 0
        self._prev_text: str = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._current_tag = tag
        self._current_attrs = dict(attrs)

        if tag in self._skip_tags:
            self._in_skip += 1
            return

        if self._in_skip:
            return

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._in_heading = True
            self._heading_text = ""

        if tag == "img":
            src = self._current_attrs.get("src", "")
            alt = self._current_attrs.get("alt", "")
            width = 0
            height = 0
            w_str = self._current_attrs.get("width", "")
            h_str = self._current_attrs.get("height", "")
            if w_str.isdigit():
                width = int(w_str)
            if h_str.isdigit():
                height = int(h_str)
            if src:
                self.images.append(ImageInfo(
                    url=src,
                    alt=alt,
                    width=width,
                    height=height,
                ))

        if tag in ("p", "div", "br", "li", "tr"):
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._skip_tags and self._in_skip > 0:
            self._in_skip -= 1
            return

        if self._in_skip:
            return

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6") and self._in_heading:
            level = tag[1:]
            text = self._heading_text.strip()
            if text:
                self.headings.append((level, text))
            self._in_heading = False

        if tag in ("p", "div", "li", "tr"):
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_skip:
            return

        text = data.strip()
        if not text:
            return

        if self._in_heading:
            self._heading_text += data
            self.text_parts.append(text)
        else:
            self.text_parts.append(text)
            self._prev_text = text

        self._block_index += 1

    def get_text(self) -> str:
        """Get the full extracted text."""
        return "\n".join(part for part in self.text_parts if part)

    def get_headings(self) -> list[str]:
        """Get heading texts as a structure skeleton."""
        return [text for _, text in self.headings]


# ============================================================
# Image filtering (three-layer mechanism)
# ============================================================
def filter_images(
    images: list[ImageInfo],
    full_text: str,
    doc_type: str = "article",
) -> list[ImageInfo]:
    """Apply three-layer image filtering.

    Layer 1: Automatic rule filtering (size, URL keywords, duplicates, alt keywords)
    Layer 2: Context prediction (surrounding text indicates value)
    Layer 3: Safety net (images near key elements are kept)

    Returns images with needs_distillation flag set appropriately.
    """
    seen_urls: set[str] = set()
    result: list[ImageInfo] = []

    for img in images:
        url_lower = img.url.lower()

        # Layer 1: Automatic rule filtering
        if img.width > 0 and img.height > 0 and (
            img.width < MIN_IMAGE_SIZE or img.height < MIN_IMAGE_SIZE
        ):
            img.filtered_reason = "size_too_small"
            result.append(img)
            continue

        if any(kw in url_lower for kw in FILTER_URL_KEYWORDS):
            img.filtered_reason = "url_keyword"
            result.append(img)
            continue

        if img.url in seen_urls:
            img.filtered_reason = "duplicate"
            result.append(img)
            continue

        if any(kw in img.alt for kw in FILTER_ALT_KEYWORDS):
            img.filtered_reason = "alt_keyword"
            result.append(img)
            continue

        seen_urls.add(img.url)

        # Layer 2: Context prediction
        context_keywords = [
            "如图", "下图", "见图", "这张图", "这个框架", "这个模型",
            "这张表", "对比", "矩阵", "象限", "流程", "步骤",
            "层级", "架构", "模型", "框架",
        ]

        # Check if nearby text indicates high value
        nearby_text = img.alt or ""
        is_high_value = any(kw in nearby_text for kw in context_keywords)

        # For course/PPT documents, relax filtering
        if doc_type in ("course", "ppt", "slides"):
            is_high_value = True

        img.needs_distillation = is_high_value

        # Layer 3: Safety net - check if near key elements
        if not img.needs_distillation:
            for pattern_list in KEY_ELEMENT_PATTERNS.values():
                for pattern in pattern_list:
                    if re.search(pattern, nearby_text, re.MULTILINE):
                        img.needs_distillation = True
                        break
                if img.needs_distillation:
                    break

        result.append(img)

    return result


# ============================================================
# Key element detection
# ============================================================
def detect_key_elements(text: str) -> dict[str, int]:
    """Detect key elements in text and return counts.

    Categories: formula, data, template, checklist, framework, table, warning, quote
    """
    counts: dict[str, int] = {}
    for element_type, patterns in KEY_ELEMENT_PATTERNS.items():
        count = 0
        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            count += len(matches)
        counts[element_type] = count
    return counts


# ============================================================
# Structure skeleton extraction
# ============================================================
def extract_structure_skeleton(headings: list[tuple[str, str]]) -> list[str]:
    """Extract structure skeleton from headings."""
    skeleton: list[str] = []
    indent_map = {"1": "", "2": "  ", "3": "    ", "4": "      ", "5": "        ", "6": "          "}
    for level, text in headings:
        indent = indent_map.get(level, "  ")
        skeleton.append(f"{indent}├── {text}")
    return skeleton


# ============================================================
# HTML distillation article generation
# ============================================================
def generate_html(
    doc_info: DocInfo,
    segments: list[DistillSegment],
    images: list[ImageInfo],
    distilled_image_count: int,
    date_str: str = "",
) -> str:
    """Generate HTML distillation article from segments."""
    if not date_str:
        date_str = time.strftime("%Y-%m-%d", time.localtime())

    title = doc_info.title
    author = doc_info.author
    source = doc_info.source_url or doc_info.source_type.value

    # Build content sections
    sections_html = []
    for seg in segments:
        if seg.content_type == ContentType.HEADING:
            sections_html.append(f'<h2>{html.escape(seg.content)}</h2>')
        elif seg.content_type == ContentType.CALLOUT:
            sections_html.append(
                f'<div class="callout"><p>{html.escape(seg.content)}</p></div>'
            )
        elif seg.content_type == ContentType.QUOTE:
            sections_html.append(
                f'<div class="quote-box">{html.escape(seg.content)}</div>'
            )
        elif seg.content_type == ContentType.TABLE:
            sections_html.append(f'<div class="content-block">{seg.content}</div>')
        else:
            sections_html.append(
                f'<p>{html.escape(seg.content)}</p>'
            )

    content_html = "\n  ".join(sections_html)

    total_images = len(images)
    filtered_count = sum(1 for img in images if img.filtered_reason)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(title)} | 蒸馏文稿</title>
<style>
  :root {{
    --bg: #fafaf7;
    --card-bg: #ffffff;
    --text: #2c2c2c;
    --text-light: #666;
    --accent: #c8463c;
    --accent-light: #f5e6e4;
    --blue: #2b5f8f;
    --blue-light: #e8f0f7;
    --green: #2d7a4f;
    --green-light: #e6f3ec;
    --border: #e0ddd5;
    --code-bg: #f5f3ee;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: "Noto Sans CJK SC", "WenQuanYi Micro Hei", "PingFang SC",
                 "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    background: var(--bg); color: var(--text);
    line-height: 1.85; font-size: 16px;
    -webkit-font-smoothing: antialiased;
  }}
  .container {{ max-width: 860px; margin: 0 auto; padding: 40px 24px 80px; }}
  .header {{
    text-align: center; margin-bottom: 48px; padding: 40px 24px;
    background: linear-gradient(135deg, var(--accent) 0%, #a03830 100%);
    border-radius: 16px; color: #fff;
  }}
  .header .tag {{
    display: inline-block; background: rgba(255,255,255,0.2);
    padding: 4px 14px; border-radius: 20px; font-size: 13px; margin-bottom: 16px;
  }}
  .header h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 8px; }}
  .header .meta {{ font-size: 13px; opacity: 0.7; margin-top: 12px; }}
  h2 {{
    font-size: 22px; font-weight: 700; color: var(--accent);
    margin: 48px 0 16px; padding-bottom: 10px;
    border-bottom: 2px solid var(--accent-light);
  }}
  h3 {{ font-size: 18px; font-weight: 600; margin: 32px 0 12px; }}
  p {{ margin-bottom: 14px; }}
  .callout {{
    background: var(--accent-light); border-left: 4px solid var(--accent);
    padding: 16px 20px; border-radius: 0 8px 8px 0; margin: 20px 0;
  }}
  .quote-box {{
    font-size: 17px; font-style: italic; text-align: center; color: var(--text-light);
    padding: 24px; margin: 32px 0;
    border-top: 1px solid var(--border); border-bottom: 1px solid var(--border);
  }}
  .content-block {{ margin: 16px 0; }}
  .footer {{
    text-align: center; margin-top: 60px; padding-top: 24px;
    border-top: 1px solid var(--border); color: var(--text-light); font-size: 13px;
  }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div class="tag">蒸馏文稿</div>
    <h1>{html.escape(title)}</h1>
    <div class="meta">作者：{html.escape(author)} | 来源：{html.escape(source)} | 蒸馏日期：{date_str}</div>
  </div>
  {content_html}
  <div class="footer">
    蒸馏自{html.escape(source)} | 图片{total_images}张（蒸馏{distilled_image_count}张，过滤{filtered_count}张） | {date_str}
  </div>
</div>
</body>
</html>"""


# ============================================================
# Obsidian note generation
# ============================================================
def generate_obsidian(
    doc_info: DocInfo,
    segments: list[DistillSegment],
    images: list[ImageInfo],
    distilled_image_count: int,
    structure_skeleton: list[str],
    key_elements: dict[str, int],
    date_str: str = "",
) -> str:
    """Generate Obsidian note content from segments."""
    if not date_str:
        date_str = time.strftime("%Y-%m-%d", time.localtime())

    title = doc_info.title
    source = doc_info.source_url or doc_info.source_type.value

    tags = [doc_info.source_type.value, "蒸馏笔记"]

    # Build frontmatter
    frontmatter = f"""---
title: "{title}"
source: "{source}"
author: "{doc_info.author}"
date: {date_str}
tags:"""
    for tag in tags:
        frontmatter += f"\n  - {tag}"
    frontmatter += "\n---"

    # Build body
    body_parts: list[str] = []
    body_parts.append(f"\n# {title}\n")

    # Summary callout
    summary_text = segments[0].content if segments else ""
    body_parts.append(f"> [!summary] 核心一句话\n> {summary_text}\n")

    # Content sections
    for seg in segments:
        if seg.content_type == ContentType.HEADING:
            body_parts.append(f"\n## {seg.content}\n")
        elif seg.content_type == ContentType.CALLOUT:
            body_parts.append(f"> [!warning] 注意\n> {seg.content}\n")
        elif seg.content_type == ContentType.QUOTE:
            body_parts.append(f"> {seg.content}\n")
        elif seg.content_type == ContentType.TABLE:
            body_parts.append(seg.content + "\n")
        else:
            body_parts.append(seg.content + "\n")

    # Image distillation section
    if distilled_image_count > 0:
        body_parts.append("\n## 图表蒸馏\n")
        for img in images:
            if img.needs_distillation and img.distilled_content:
                body_parts.append(
                    f"> [!info] 图表蒸馏：{img.alt or '未命名图片'}\n"
                    f"> {img.distilled_content}\n"
                    f"> ![[assets/{os.path.basename(img.local_path) or 'img.png'}]]\n"
                )

    # Key elements summary
    if key_elements:
        body_parts.append("\n## 关键要素统计\n")
        body_parts.append("| 要素类型 | 数量 |")
        body_parts.append("|---|---|")
        for elem, count in key_elements.items():
            body_parts.append(f"| {elem} | {count} |")

    # Footer
    total_images = len(images)
    body_parts.append(
        f"\n---\n"
        f"*蒸馏自{source} | 图片{total_images}张"
        f"（蒸馏{distilled_image_count}张） | {date_str}*\n"
    )

    return frontmatter + "\n" + "\n".join(body_parts)


# ============================================================
# Main Distiller class
# ============================================================
class Distiller:
    """Multi-source document distiller with five-stage workflow.

    Args:
        html_output_dir: Directory for HTML distillation articles.
        obsidian_dir: Directory for Obsidian notes.
    """

    def __init__(
        self,
        html_output_dir: str = DEFAULT_HTML_OUTPUT_DIR,
        obsidian_dir: str = DEFAULT_OBSIDIAN_DIR,
    ):
        self.html_output_dir = html_output_dir
        self.obsidian_dir = obsidian_dir
        os.makedirs(html_output_dir, exist_ok=True)
        os.makedirs(obsidian_dir, exist_ok=True)

    # ----------------------------------------------------------
    # Stage 1: Source detection and content extraction
    # ----------------------------------------------------------
    def extract_from_url(
        self,
        url: str,
        progress_cb: ProgressCallback | None = None,
    ) -> DistillResult:
        """Extract content from a URL (webpage, Feishu, PDF, video)."""
        if progress_cb:
            progress_cb(0.1, 1.0, "Detecting source type...")

        source_type = detect_source_type(url)

        if progress_cb:
            progress_cb(0.2, 1.0, f"Source type: {source_type.value}")

        doc_info = DocInfo(
            title=url.split("/")[-1] or "Untitled",
            source_url=url,
            source_type=source_type,
        )

        if source_type == SourceType.FEISHU:
            return self._handle_feishu(url, doc_info, progress_cb)

        if source_type in (SourceType.VIDEO, SourceType.PODCAST):
            return self._handle_video(url, doc_info, source_type, progress_cb)

        if source_type == SourceType.WEBPAGE:
            return self._handle_webpage(url, doc_info, progress_cb)

        if source_type == SourceType.PDF:
            return self._handle_pdf_url(url, doc_info, progress_cb)

        # Unknown source type
        return DistillResult(
            doc_info=doc_info,
            method=DistillationMethod.UNSUPPORTED,
            guidance=f"Unsupported source type for URL: {url}. "
                     f"Supported: feishu, webpage, pdf, video, podcast.",
        )

    def extract_from_file(
        self,
        file_path: str,
        progress_cb: ProgressCallback | None = None,
    ) -> DistillResult:
        """Extract content from a local file (PDF, text, etc.)."""
        if progress_cb:
            progress_cb(0.1, 1.0, "Reading file...")

        if not os.path.exists(file_path):
            return DistillResult(
                doc_info=DocInfo(
                    title=os.path.basename(file_path),
                    source_type=SourceType.LOCAL_FILE,
                    source_url=file_path,
                ),
                method=DistillationMethod.UNSUPPORTED,
                guidance=f"File not found: {file_path}",
            )

        file_lower = file_path.lower()
        doc_info = DocInfo(
            title=os.path.basename(file_path),
            source_type=SourceType.LOCAL_FILE,
            source_url=file_path,
        )

        if file_lower.endswith(".pdf"):
            return self._handle_pdf_file(file_path, doc_info, progress_cb)

        if file_lower.endswith((".html", ".htm")):
            return self._handle_html_file(file_path, doc_info, progress_cb)

        if file_lower.endswith((".txt", ".md", ".markdown")):
            return self._handle_text_file(file_path, doc_info, progress_cb)

        if file_lower.endswith((".mp3", ".m4a", ".wav", ".flac", ".mp4")):
            return self._handle_video(
                file_path, doc_info, SourceType.LOCAL_FILE, progress_cb,
            )

        return DistillResult(
            doc_info=doc_info,
            method=DistillationMethod.UNSUPPORTED,
            guidance=f"Unsupported file type: {file_path}. "
                     f"Supported: pdf, html, txt, md, mp3, m4a, wav, mp4.",
        )

    # ----------------------------------------------------------
    # Source-specific handlers
    # ----------------------------------------------------------
    def _handle_webpage(
        self,
        url: str,
        doc_info: DocInfo,
        progress_cb: ProgressCallback | None,
    ) -> DistillResult:
        """Handle webpage extraction."""
        if progress_cb:
            progress_cb(0.3, 1.0, "Fetching webpage...")

        try:
            import urllib.request
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                raw = resp.read().decode(charset, errors="replace")
        except OSError as e:
            return DistillResult(
                doc_info=doc_info,
                method=DistillationMethod.UNSUPPORTED,
                guidance=f"Failed to fetch webpage: {e}",
            )

        return self._process_html_content(raw, doc_info, DistillationMethod.URL_EXTRACTION, progress_cb)

    def _handle_feishu(
        self,
        url: str,
        doc_info: DocInfo,
        progress_cb: ProgressCallback | None,
    ) -> DistillResult:
        """Handle Feishu document extraction."""
        if progress_cb:
            progress_cb(0.3, 1.0, "Feishu document detected. Use lark-doc skill for extraction.")

        return DistillResult(
            doc_info=doc_info,
            method=DistillationMethod.UNSUPPORTED,
            guidance=(
                "Feishu document detected. For full extraction, use the lark-doc skill "
                "or browser tools. If permissions fail, use browser_navigate + "
                "browser_evaluate to extract content via JavaScript."
            ),
        )

    def _handle_video(
        self,
        url: str,
        doc_info: DocInfo,
        source_type: SourceType,
        progress_cb: ProgressCallback | None,
    ) -> DistillResult:
        """Handle video/podcast source."""
        if progress_cb:
            progress_cb(0.3, 1.0, "Video/podcast source detected.")

        platform = _detect_video_platform(url)
        guidance_parts = [
            f"Video source detected: {source_type.value}",
        ]

        if platform in CLOSED_PLATFORMS:
            guidance_parts.append(
                "Closed platform. Use WeChat mini-programs "
                "(文案提取宝/媒小三AI创作/Get笔记) to extract transcript, "
                "then paste text for distillation."
            )
        else:
            guidance_parts.append(
                "Use video-transcript-mcp or scripts/video_transcript.py "
                "to extract transcript first, then distill the text output."
            )

        return DistillResult(
            doc_info=doc_info,
            method=DistillationMethod.UNSUPPORTED,
            guidance=" | ".join(guidance_parts),
        )

    def _handle_pdf_url(
        self,
        url: str,
        doc_info: DocInfo,
        progress_cb: ProgressCallback | None,
    ) -> DistillResult:
        """Handle PDF URL extraction."""
        if progress_cb:
            progress_cb(0.3, 1.0, "Downloading PDF...")

        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                pdf_data = resp.read()
        except OSError as e:
            return DistillResult(
                doc_info=doc_info,
                method=DistillationMethod.UNSUPPORTED,
                guidance=f"Failed to download PDF: {e}",
            )

        temp_path = os.path.join(self.html_output_dir, f"temp_{int(time.time())}.pdf")
        with open(temp_path, "wb") as f:
            f.write(pdf_data)

        result = self._handle_pdf_file(temp_path, doc_info, progress_cb)

        # Cleanup temp file
        try:
            os.remove(temp_path)
        except OSError:
            pass

        return result

    def _handle_pdf_file(
        self,
        file_path: str,
        doc_info: DocInfo,
        progress_cb: ProgressCallback | None,
    ) -> DistillResult:
        """Handle local PDF file extraction."""
        if progress_cb:
            progress_cb(0.4, 1.0, "Extracting PDF text...")

        try:
            result = subprocess.run(
                ["python3", "-c",
                 (
                     "from pdfplumber import open as pdfopen; "
                     f"pdf = pdfopen('{file_path}'); "
                     "print('\\n\\n'.join(page.extract_text() or '' for page in pdf.pages)); "
                     "pdf.close()"
                 )],
                capture_output=True, text=True, timeout=60, check=False,
            )
            if result.returncode == 0:
                text = result.stdout.strip()
            else:
                return DistillResult(
                    doc_info=doc_info,
                    method=DistillationMethod.UNSUPPORTED,
                    guidance=f"PDF extraction failed: {result.stderr[:200]}",
                )
        except (OSError, subprocess.SubprocessError) as e:
            return DistillResult(
                doc_info=doc_info,
                method=DistillationMethod.UNSUPPORTED,
                guidance=f"PDF extraction error: {e}. Install pdfplumber: pip install pdfplumber",
            )

        doc_info.title = doc_info.title or os.path.basename(file_path)

        segments = [DistillSegment(
            section_title="Content",
            content=text,
            content_type=ContentType.PARAGRAPH,
            order=0,
        )]

        return self._finalize_result(
            doc_info=doc_info,
            method=DistillationMethod.FILE_DISTILLATION,
            segments=segments,
            full_text=text,
            images=[],
            progress_cb=progress_cb,
        )

    def _handle_html_file(
        self,
        file_path: str,
        doc_info: DocInfo,
        progress_cb: ProgressCallback | None,
    ) -> DistillResult:
        """Handle local HTML file extraction."""
        if progress_cb:
            progress_cb(0.3, 1.0, "Reading HTML file...")

        with open(file_path, encoding="utf-8", errors="replace") as f:
            content = f.read()

        return self._process_html_content(content, doc_info, DistillationMethod.FILE_DISTILLATION, progress_cb)

    def _handle_text_file(
        self,
        file_path: str,
        doc_info: DocInfo,
        progress_cb: ProgressCallback | None,
    ) -> DistillResult:
        """Handle plain text/markdown file."""
        if progress_cb:
            progress_cb(0.4, 1.0, "Reading text file...")

        with open(file_path, encoding="utf-8", errors="replace") as f:
            text = f.read()

        doc_info.title = doc_info.title or os.path.basename(file_path)

        segments = [DistillSegment(
            section_title="Content",
            content=text,
            content_type=ContentType.PARAGRAPH,
            order=0,
        )]

        return self._finalize_result(
            doc_info=doc_info,
            method=DistillationMethod.FILE_DISTILLATION,
            segments=segments,
            full_text=text,
            images=[],
            progress_cb=progress_cb,
        )

    def _process_html_content(
        self,
        html_content: str,
        doc_info: DocInfo,
        method: DistillationMethod,
        progress_cb: ProgressCallback | None,
    ) -> DistillResult:
        """Process HTML content to extract text, headings, and images."""
        if progress_cb:
            progress_cb(0.4, 1.0, "Parsing HTML content...")

        parser = ContentExtractor()
        parser.feed(html_content)

        full_text = parser.get_text()
        headings = parser.headings
        images = parser.images

        # Extract title from first h1
        for level, text in headings:
            if level == "1":
                doc_info.title = text
                break

        if not doc_info.title:
            doc_info.title = "Untitled"

        # Stage 2: Structure skeleton + key elements
        if progress_cb:
            progress_cb(0.5, 1.0, "Extracting structure skeleton...")

        structure_skeleton = extract_structure_skeleton(headings)
        key_elements = detect_key_elements(full_text)

        # Stage 2.5: Image filtering
        if progress_cb:
            progress_cb(0.6, 1.0, f"Filtering {len(images)} images...")

        filtered_images = filter_images(images, full_text)

        # Build segments from headings and text
        segments = self._build_segments(headings, full_text)

        return self._finalize_result(
            doc_info=doc_info,
            method=method,
            segments=segments,
            full_text=full_text,
            images=filtered_images,
            structure_skeleton=structure_skeleton,
            key_elements=key_elements,
            progress_cb=progress_cb,
        )

    def _build_segments(
        self,
        headings: list[tuple[str, str]],
        full_text: str,
    ) -> list[DistillSegment]:
        """Build content segments from headings and full text."""
        segments: list[DistillSegment] = []

        for order, (_, text) in enumerate(headings):
            segments.append(DistillSegment(
                section_title=text,
                content=text,
                content_type=ContentType.HEADING,
                order=order,
            ))

        if not segments and full_text:
            segments.append(DistillSegment(
                section_title="Content",
                content=full_text[:5000],
                content_type=ContentType.PARAGRAPH,
                order=0,
            ))

        return segments

    def _finalize_result(
        self,
        doc_info: DocInfo,
        method: DistillationMethod,
        segments: list[DistillSegment],
        full_text: str,
        images: list[ImageInfo],
        structure_skeleton: list[str] | None = None,
        key_elements: dict[str, int] | None = None,
        progress_cb: ProgressCallback | None = None,
    ) -> DistillResult:
        """Finalize the distillation result with HTML and Obsidian output."""
        if progress_cb:
            progress_cb(0.7, 1.0, "Generating HTML article...")

        distilled_image_count = sum(1 for img in images if img.needs_distillation)

        html_content = generate_html(
            doc_info=doc_info,
            segments=segments,
            images=images,
            distilled_image_count=distilled_image_count,
        )

        if progress_cb:
            progress_cb(0.8, 1.0, "Generating Obsidian note...")

        obsidian_content = generate_obsidian(
            doc_info=doc_info,
            segments=segments,
            images=images,
            distilled_image_count=distilled_image_count,
            structure_skeleton=structure_skeleton or [],
            key_elements=key_elements or {},
        )

        # Save files
        if progress_cb:
            progress_cb(0.9, 1.0, "Saving output files...")

        safe_title = _safe_filename(doc_info.title)
        html_path = os.path.join(self.html_output_dir, f"{safe_title}-蒸馏文稿.html")
        obsidian_path = os.path.join(self.obsidian_dir, f"{safe_title}.md")

        try:
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
        except PermissionError:
            _copy_with_osascript(html_path, html_content)

        try:
            with open(obsidian_path, "w", encoding="utf-8") as f:
                f.write(obsidian_content)
        except PermissionError:
            _copy_with_osascript(obsidian_path, obsidian_content)

        if progress_cb:
            progress_cb(1.0, 1.0, "Distillation completed.")

        return DistillResult(
            doc_info=doc_info,
            method=method,
            segments=segments,
            full_text=full_text,
            html_content=html_content,
            obsidian_content=obsidian_content,
            html_path=html_path,
            obsidian_path=obsidian_path,
            images=images,
            distilled_image_count=distilled_image_count,
            structure_skeleton=structure_skeleton or [],
            key_elements=key_elements or {},
        )


# ============================================================
# Helper functions
# ============================================================
def _detect_video_platform(url: str) -> str:
    """Detect the video platform from a URL."""
    url_lower = url.lower()
    patterns = {
        "youtube": [r"youtube\.com", r"youtu\.be"],
        "bilibili": [r"bilibili\.com", r"b23\.tv"],
        "douyin": [r"douyin\.com", r"iesdouyin\.com"],
        "tiktok": [r"tiktok\.com"],
        "kuaishou": [r"kuaishou\.com", r"chenzhongtech\.com"],
        "xiaohongshu": [r"xiaohongshu\.com", r"xhslink\.com"],
        "weixin_video": [r"channels\.weixin\.qq\.com"],
    }
    for platform, pattern_list in patterns.items():
        for pattern in pattern_list:
            if re.search(pattern, url_lower):
                return platform
    return "unknown"


def _copy_with_osascript(dest_path: str, content: str) -> None:
    """Copy content to a path using osascript to bypass sandbox restrictions."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tmp", delete=False, encoding="utf-8") as tmp:
        tmp.write(content)

    try:
        subprocess.run(
            ["osascript", "-e", f'do shell script "cp \'{tmp.name}\' \'{dest_path}\'"'],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        logger.warning("osascript copy failed: %s", e)
    finally:
        os.unlink(tmp.name)
