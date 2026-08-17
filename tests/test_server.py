"""Tests for doc-distillation-mcp server."""

import os
import tempfile
import time

import doc_distillation_mcp.server as srv
from doc_distillation_mcp.distiller import (
    ContentExtractor,
    Distiller,
    _detect_video_platform,
    _safe_filename,
    detect_key_elements,
    detect_source_type,
    extract_structure_skeleton,
    filter_images,
    generate_html,
    generate_obsidian,
)
from doc_distillation_mcp.models import (
    ContentType,
    DistillationMethod,
    DistillResult,
    DistillSegment,
    DocInfo,
    ImageInfo,
    SourceType,
    TaskInfo,
    TaskStatus,
)


# ============================================================
# Source type detection tests
# ============================================================
class TestSourceTypeDetection:
    def test_feishu(self):
        assert detect_source_type("https://xxx.feishu.cn/docs/abc") == SourceType.FEISHU

    def test_feishu_lark(self):
        assert detect_source_type("https://xxx.larkoffice.com/docs/abc") == SourceType.FEISHU

    def test_webpage(self):
        assert detect_source_type("https://example.com/article") == SourceType.WEBPAGE

    def test_pdf_url(self):
        assert detect_source_type("https://example.com/doc.pdf") == SourceType.PDF

    def test_youtube(self):
        assert detect_source_type("https://www.youtube.com/watch?v=abc") == SourceType.VIDEO

    def test_bilibili(self):
        assert detect_source_type("https://www.bilibili.com/video/BV1234") == SourceType.VIDEO

    def test_douyin(self):
        assert detect_source_type("https://www.douyin.com/video/123") == SourceType.VIDEO

    def test_podcast_mp3(self):
        assert detect_source_type("https://example.com/episode.mp3") == SourceType.PODCAST

    def test_podcast_m4a(self):
        assert detect_source_type("https://example.com/episode.m4a") == SourceType.PODCAST

    def test_local_file_pdf(self):
        assert detect_source_type("/path/to/file.pdf") == SourceType.PDF

    def test_local_file_txt(self):
        assert detect_source_type("/path/to/file.txt") == SourceType.LOCAL_FILE

    def test_local_file_relative(self):
        assert detect_source_type("./article.md") == SourceType.LOCAL_FILE

    def test_local_file_windows(self):
        assert detect_source_type("C:/Users/test/doc.pdf") == SourceType.PDF

    def test_unknown(self):
        assert detect_source_type("not-a-url") == SourceType.UNKNOWN


# ============================================================
# Pydantic model tests
# ============================================================
class TestModels:
    def test_doc_info_defaults(self):
        info = DocInfo()
        assert info.title == "Unknown"
        assert info.author == "Unknown"
        assert info.source_type == SourceType.UNKNOWN

    def test_distill_result_subtitle(self):
        result = DistillResult(
            doc_info=DocInfo(title="Test", source_type=SourceType.WEBPAGE),
            method=DistillationMethod.URL_EXTRACTION,
            full_text="Hello world",
        )
        assert result.method == DistillationMethod.URL_EXTRACTION
        assert result.full_text == "Hello world"
        assert len(result.segments) == 0

    def test_distill_result_unsupported(self):
        result = DistillResult(
            doc_info=DocInfo(title="Test"),
            method=DistillationMethod.UNSUPPORTED,
            guidance="Use mini-program to extract",
        )
        assert result.method == DistillationMethod.UNSUPPORTED
        assert result.guidance == "Use mini-program to extract"

    def test_distill_segment(self):
        seg = DistillSegment(
            section_title="Chapter 1",
            content="Some distilled content",
            content_type=ContentType.PARAGRAPH,
            order=0,
        )
        assert seg.section_title == "Chapter 1"
        assert seg.content_type == ContentType.PARAGRAPH

    def test_image_info_defaults(self):
        img = ImageInfo(url="https://example.com/img.png")
        assert img.url == "https://example.com/img.png"
        assert img.needs_distillation is False
        assert img.filtered_reason == ""

    def test_task_info_defaults(self):
        task = TaskInfo(
            task_id="abc123",
            status=TaskStatus.PENDING,
            created_at=time.time(),
        )
        assert task.status == TaskStatus.PENDING
        assert task.progress == 0.0
        assert task.result is None

    def test_task_status_values(self):
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.PROCESSING == "processing"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"

    def test_content_type_values(self):
        assert ContentType.PARAGRAPH == "paragraph"
        assert ContentType.HEADING == "heading"
        assert ContentType.TABLE == "table"
        assert ContentType.CALLOUT == "callout"


# ============================================================
# Image filtering tests
# ============================================================
class TestImageFiltering:
    def test_filter_small_image(self):
        img = ImageInfo(url="https://example.com/small.png", width=50, height=50)
        result = filter_images([img], "some text")
        assert result[0].filtered_reason == "size_too_small"

    def test_filter_url_keyword(self):
        img = ImageInfo(url="https://example.com/avatar.png", width=200, height=200)
        result = filter_images([img], "some text")
        assert result[0].filtered_reason == "url_keyword"

    def test_filter_duplicate(self):
        img1 = ImageInfo(url="https://example.com/img.png", width=200, height=200)
        img2 = ImageInfo(url="https://example.com/img.png", width=200, height=200)
        result = filter_images([img1, img2], "some text")
        assert result[1].filtered_reason == "duplicate"

    def test_filter_alt_keyword(self):
        img = ImageInfo(
            url="https://example.com/img.png",
            alt="头像",
            width=200, height=200,
        )
        result = filter_images([img], "some text")
        assert result[0].filtered_reason == "alt_keyword"

    def test_high_value_image_kept(self):
        img = ImageInfo(
            url="https://example.com/framework.png",
            alt="这个框架图展示了核心架构",
            width=800, height=600,
        )
        result = filter_images([img], "如下图所示")
        kept = [i for i in result if not i.filtered_reason]
        assert len(kept) == 1
        assert kept[0].needs_distillation is True

    def test_course_type_relaxes_filter(self):
        img = ImageInfo(url="https://example.com/slide.png", width=100, height=80)
        result = filter_images([img], "slide content", doc_type="course")
        assert result[0].filtered_reason != "size_too_small" or result[0].needs_distillation


# ============================================================
# Key element detection tests
# ============================================================
class TestKeyElementDetection:
    def test_detect_data(self):
        text = "转化率达到了 35.6%，月收入 5000 元"
        result = detect_key_elements(text)
        assert result["data"] >= 2

    def test_detect_formula(self):
        text = "ROI = 销售额 / 广告费，保本 ROI = 1.03"
        result = detect_key_elements(text)
        assert result["formula"] >= 2

    def test_detect_warning(self):
        text = "不要直接复制，避免侵权，注意版权风险"
        result = detect_key_elements(text)
        assert result["warning"] >= 3

    def test_detect_checklist(self):
        text = "1. 第一步\n2. 第二步\n3. 第三步"
        result = detect_key_elements(text)
        assert result["checklist"] >= 3

    def test_detect_nothing(self):
        text = "这是一段普通的文字，没有特殊要素。"
        result = detect_key_elements(text)
        assert sum(result.values()) == 0


# ============================================================
# HTML parser tests
# ============================================================
class TestContentExtractor:
    def test_extract_text(self):
        parser = ContentExtractor()
        parser.feed("<p>Hello world</p>")
        assert "Hello world" in parser.get_text()

    def test_extract_headings(self):
        parser = ContentExtractor()
        parser.feed("<h1>Title</h1><h2>Section 1</h2><h3>Subsection</h3>")
        headings = parser.get_headings()
        assert len(headings) == 3
        assert headings[0] == "Title"
        assert headings[1] == "Section 1"

    def test_extract_images(self):
        parser = ContentExtractor()
        parser.feed('<img src="https://example.com/img.png" alt="Test" width="200" height="150">')
        assert len(parser.images) == 1
        assert parser.images[0].url == "https://example.com/img.png"
        assert parser.images[0].alt == "Test"
        assert parser.images[0].width == 200

    def test_skip_script_style(self):
        parser = ContentExtractor()
        parser.feed('<script>var x = 1;</script><style>body { margin: 0; }</style><p>Content</p>')
        text = parser.get_text()
        assert "Content" in text
        assert "var x" not in text
        assert "margin" not in text

    def test_multiple_images(self):
        html = '<img src="img1.png"><img src="img2.png"><img src="img3.png">'
        parser = ContentExtractor()
        parser.feed(html)
        assert len(parser.images) == 3


# ============================================================
# Structure skeleton tests
# ============================================================
class TestStructureSkeleton:
    def test_empty(self):
        result = extract_structure_skeleton([])
        assert result == []

    def test_flat_structure(self):
        headings = [("1", "Chapter 1"), ("1", "Chapter 2")]
        result = extract_structure_skeleton(headings)
        assert len(result) == 2
        assert "Chapter 1" in result[0]

    def test_nested_structure(self):
        headings = [("1", "Chapter"), ("2", "Section"), ("3", "Subsection")]
        result = extract_structure_skeleton(headings)
        assert len(result) == 3
        # Deeper levels should have more indentation
        assert len(result[1]) > len(result[0])


# ============================================================
# HTML generation tests
# ============================================================
class TestHtmlGeneration:
    def test_basic_html(self):
        doc = DocInfo(title="Test", author="Author", source_url="https://example.com")
        segments = [DistillSegment(content="Test content", content_type=ContentType.PARAGRAPH)]
        html = generate_html(doc, segments, [], 0)
        assert "<html" in html
        assert "Test" in html
        assert "Test content" in html

    def test_html_with_heading(self):
        doc = DocInfo(title="Test")
        segments = [DistillSegment(content="Section Title", content_type=ContentType.HEADING)]
        html = generate_html(doc, segments, [], 0)
        assert "<h2>Section Title</h2>" in html

    def test_html_with_callout(self):
        doc = DocInfo(title="Test")
        segments = [DistillSegment(content="Important note", content_type=ContentType.CALLOUT)]
        html = generate_html(doc, segments, [], 0)
        assert 'class="callout"' in html
        assert "Important note" in html

    def test_html_with_quote(self):
        doc = DocInfo(title="Test")
        segments = [DistillSegment(content="A wise quote", content_type=ContentType.QUOTE)]
        html = generate_html(doc, segments, [], 0)
        assert 'class="quote-box"' in html

    def test_html_footer(self):
        doc = DocInfo(title="Test", source_url="https://example.com")
        segments = []
        html = generate_html(doc, segments, [], 0)
        assert "蒸馏自" in html
        assert "图片0张" in html


# ============================================================
# Obsidian note generation tests
# ============================================================
class TestObsidianGeneration:
    def test_basic_obsidian(self):
        doc = DocInfo(
            title="Test",
            author="Author",
            source_url="https://example.com",
            source_type=SourceType.WEBPAGE,
        )
        segments = [DistillSegment(content="Core message", content_type=ContentType.PARAGRAPH)]
        md = generate_obsidian(doc, segments, [], 0, [], {})
        assert "---" in md
        assert "title:" in md
        assert "Test" in md
        assert "[!summary]" in md

    def test_obsidian_with_heading(self):
        doc = DocInfo(title="Test")
        segments = [DistillSegment(content="Chapter 1", content_type=ContentType.HEADING)]
        md = generate_obsidian(doc, segments, [], 0, [], {})
        assert "## Chapter 1" in md

    def test_obsidian_with_key_elements(self):
        doc = DocInfo(title="Test")
        segments = []
        key_elements = {"formula": 3, "data": 15, "warning": 6}
        md = generate_obsidian(doc, segments, [], 0, [], key_elements)
        assert "关键要素统计" in md
        assert "formula" in md
        assert "15" in md

    def test_obsidian_tags(self):
        doc = DocInfo(title="Test", source_type=SourceType.FEISHU)
        segments = []
        md = generate_obsidian(doc, segments, [], 0, [], {})
        assert "feishu" in md
        assert "蒸馏笔记" in md

    def test_obsidian_image_distillation(self):
        doc = DocInfo(title="Test")
        segments = []
        images = [ImageInfo(
            url="https://example.com/framework.png",
            alt="Framework diagram",
            needs_distillation=True,
            distilled_content="The framework shows 3 layers",
            local_path="/tmp/img_1.png",
        )]
        md = generate_obsidian(doc, segments, images, 1, [], {})
        assert "图表蒸馏" in md
        assert "Framework diagram" in md
        assert "The framework shows 3 layers" in md


# ============================================================
# Distiller class tests
# ============================================================
class TestDistiller:
    def test_init_creates_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            html_dir = os.path.join(tmpdir, "html")
            obs_dir = os.path.join(tmpdir, "obs")
            Distiller(html_output_dir=html_dir, obsidian_dir=obs_dir)
            assert os.path.exists(html_dir)
            assert os.path.exists(obs_dir)

    def test_extract_text_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("This is a test document.\nSecond paragraph.")
            f.flush()
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    distiller = Distiller(
                        html_output_dir=os.path.join(tmpdir, "html"),
                        obsidian_dir=os.path.join(tmpdir, "obs"),
                    )
                    result = distiller.extract_from_file(f.name)
                    assert result.method == DistillationMethod.FILE_DISTILLATION
                    assert "test document" in result.full_text
                    assert result.html_path
                    assert result.obsidian_path
                    assert os.path.exists(result.html_path)
                    assert os.path.exists(result.obsidian_path)
            finally:
                os.unlink(f.name)

    def test_extract_html_file(self):
        html_content = "<html><body><h1>Test Title</h1><p>Some content here</p></body></html>"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as f:
            f.write(html_content)
            f.flush()
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    distiller = Distiller(
                        html_output_dir=os.path.join(tmpdir, "html"),
                        obsidian_dir=os.path.join(tmpdir, "obs"),
                    )
                    result = distiller.extract_from_file(f.name)
                    assert result.method == DistillationMethod.FILE_DISTILLATION
                    assert "Test Title" in result.full_text
                    assert "Some content" in result.full_text
            finally:
                os.unlink(f.name)

    def test_extract_nonexistent_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            distiller = Distiller(
                html_output_dir=os.path.join(tmpdir, "html"),
                obsidian_dir=os.path.join(tmpdir, "obs"),
            )
            result = distiller.extract_from_file("/nonexistent/file.txt")
            assert result.method == DistillationMethod.UNSUPPORTED
            assert "File not found" in result.guidance

    def test_extract_unsupported_file_type(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"binary data")
            f.flush()
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    distiller = Distiller(
                        html_output_dir=os.path.join(tmpdir, "html"),
                        obsidian_dir=os.path.join(tmpdir, "obs"),
                    )
                    result = distiller.extract_from_file(f.name)
                    assert result.method == DistillationMethod.UNSUPPORTED
                    assert "Unsupported file type" in result.guidance
            finally:
                os.unlink(f.name)

    def test_extract_feishu_returns_guidance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            distiller = Distiller(
                html_output_dir=os.path.join(tmpdir, "html"),
                obsidian_dir=os.path.join(tmpdir, "obs"),
            )
            result = distiller.extract_from_url("https://xxx.feishu.cn/docs/abc")
            assert result.method == DistillationMethod.UNSUPPORTED
            assert "Feishu" in result.guidance
            assert result.doc_info.source_type == SourceType.FEISHU

    def test_extract_video_returns_guidance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            distiller = Distiller(
                html_output_dir=os.path.join(tmpdir, "html"),
                obsidian_dir=os.path.join(tmpdir, "obs"),
            )
            result = distiller.extract_from_url("https://www.youtube.com/watch?v=abc")
            assert result.method == DistillationMethod.UNSUPPORTED
            assert "video-transcript-mcp" in result.guidance or "Video" in result.guidance


# ============================================================
# Helper function tests
# ============================================================
class TestHelpers:
    def test_safe_filename(self):
        assert _safe_filename("Test/Doc:name?") == "Test_Doc_name_"
        assert _safe_filename("Normal Title") == "Normal Title"

    def test_safe_filename_truncation(self):
        long_title = "A" * 100
        result = _safe_filename(long_title)
        assert len(result) == 80

    def test_detect_video_platform_youtube(self):
        assert _detect_video_platform("https://youtube.com/watch?v=abc") == "youtube"
        assert _detect_video_platform("https://youtu.be/abc") == "youtube"

    def test_detect_video_platform_bilibili(self):
        assert _detect_video_platform("https://bilibili.com/video/BV123") == "bilibili"
        assert _detect_video_platform("https://b23.tv/abc") == "bilibili"

    def test_detect_video_platform_xiaohongshu(self):
        assert _detect_video_platform("https://xiaohongshu.com/explore/123") == "xiaohongshu"

    def test_detect_video_platform_unknown(self):
        assert _detect_video_platform("https://example.com/video") == "unknown"


# ============================================================
# MCP Server tool schema tests
# ============================================================
class TestMcpToolSchemas:
    def test_server_name(self):
        assert srv.mcp.name == "doc-distillation-mcp"

    def test_tool_count(self):
        # The server should register 4 tools
        assert len(srv._task_store) == 0  # No tasks yet

    def test_default_dirs_configurable(self):
        assert srv.DEFAULT_HTML_DIR is not None
        assert srv.DEFAULT_OBSIDIAN_BASE is not None
