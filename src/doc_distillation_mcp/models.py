"""Pydantic models for structured distillation output."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class DistillationMethod(str, Enum):
    """How the distillation was performed."""
    URL_EXTRACTION = "url_extraction"      # Content fetched from URL
    FILE_DISTILLATION = "file_distillation"  # Local file processed
    UNSUPPORTED = "unsupported"            # Source not supported (guidance returned)


class SourceType(str, Enum):
    """Type of the source document."""
    FEISHU = "feishu"
    WEBPAGE = "webpage"
    PDF = "pdf"
    VIDEO = "video"
    PODCAST = "podcast"
    LOCAL_FILE = "local_file"
    TEXT = "text"
    UNKNOWN = "unknown"


class ContentType(str, Enum):
    """Type of a content segment in the distilled output."""
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    LIST = "list"
    CALLOUT = "callout"
    IMAGE = "image"
    CODE = "code"
    QUOTE = "quote"


class ImageInfo(BaseModel):
    """Metadata about an image found in the source."""
    url: str = Field(default="", description="Image URL")
    alt: str = Field(default="", description="Alt text")
    width: int = Field(default=0, description="Image width in pixels")
    height: int = Field(default=0, description="Image height in pixels")
    local_path: str = Field(default="", description="Local file path after download")
    needs_distillation: bool = Field(default=False, description="Whether image contains text to distill")
    distilled_content: str = Field(default="", description="Extracted text content from image")
    filtered_reason: str = Field(default="", description="Why image was filtered out (if applicable)")


class DistillSegment(BaseModel):
    """A single segment of distilled content."""
    section_title: str = Field(default="", description="Section heading")
    content: str = Field(description="Distilled text content")
    content_type: ContentType = Field(
        default=ContentType.PARAGRAPH,
        description="Type of content",
    )
    order: int = Field(default=0, description="Order in document")


class DocInfo(BaseModel):
    """Metadata about the source document."""
    title: str = Field(default="Unknown", description="Document title")
    author: str = Field(default="Unknown", description="Author or creator")
    source_url: str = Field(default="", description="Source URL")
    source_type: SourceType = Field(default=SourceType.UNKNOWN, description="Source type")
    date: str = Field(default="", description="Publication or creation date")


class DistillResult(BaseModel):
    """Complete distillation result."""
    doc_info: DocInfo = Field(description="Source document metadata")
    method: DistillationMethod = Field(description="Distillation method used")
    segments: list[DistillSegment] = Field(
        default_factory=list,
        description="Distilled content segments",
    )
    full_text: str = Field(default="", description="Full distilled text")
    html_content: str = Field(
        default="",
        description="Generated HTML distillation article",
    )
    obsidian_content: str = Field(
        default="",
        description="Generated Obsidian note content",
    )
    html_path: str = Field(default="", description="Saved HTML file path")
    obsidian_path: str = Field(default="", description="Saved Obsidian note file path")
    images: list[ImageInfo] = Field(
        default_factory=list,
        description="Images found in the source",
    )
    distilled_image_count: int = Field(
        default=0,
        description="Number of images with distilled content",
    )
    structure_skeleton: list[str] = Field(
        default_factory=list,
        description="Section headings extracted from source (for integrity check)",
    )
    key_elements: dict[str, int] = Field(
        default_factory=dict,
        description="Key element counts: formula, data, template, checklist, framework, table, warning, quote",
    )
    warning: str | None = Field(default=None, description="Warning message if issues occurred")
    guidance: str | None = Field(default=None, description="Guidance for unsupported sources")


class TaskStatus(str, Enum):
    """Status of an async distillation task."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskInfo(BaseModel):
    """Status info for an async distillation task."""
    task_id: str = Field(description="Unique task identifier")
    status: TaskStatus = Field(description="Current task status")
    progress: float = Field(default=0.0, description="Progress 0.0-1.0")
    message: str = Field(default="", description="Human-readable status message")
    result: DistillResult | None = Field(default=None, description="Result if completed")
    error: str | None = Field(default=None, description="Error message if failed")
    created_at: float = Field(description="Task creation timestamp")
    completed_at: float | None = Field(default=None, description="Task completion timestamp")


class DistillListItem(BaseModel):
    """A completed distillation in the listing."""
    task_id: str = Field(description="Task identifier")
    title: str = Field(description="Document title")
    source_type: str = Field(description="Source type")
    method: DistillationMethod = Field(description="Distillation method")
    segment_count: int = Field(default=0, description="Number of distilled segments")
    image_count: int = Field(default=0, description="Total images found")
    distilled_image_count: int = Field(default=0, description="Images with distilled content")
    created_at: float = Field(description="Creation timestamp")
