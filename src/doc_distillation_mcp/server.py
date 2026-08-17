"""MCP Server for document distillation.

Provides 4 tools:
  - distill_url: Distill content from URL (Feishu, webpage, PDF, video)
  - distill_file: Distill content from local file (PDF, HTML, text, etc.)
  - get_distill_status: Poll status of async distillation task
  - list_distillations: List all completed distillations

Five-stage workflow:
  Stage 1: Source detection and content extraction
  Stage 2: Integrity safeguard (structure skeleton + key elements)
  Stage 2.5: Image filtering (three-layer mechanism)
  Stage 3: HTML distillation article generation
  Stage 4: Obsidian note generation
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Annotated

from pydantic import Field

from .distiller import (
    DEFAULT_HTML_OUTPUT_DIR,
    DEFAULT_OBSIDIAN_DIR,
    Distiller,
)
from .models import (
    DistillResult,
    TaskInfo,
    TaskStatus,
)

# MCP SDK import (compatible with v1 and v2)
try:
    from mcp.server import MCPServer
    from mcp.server.mcpserver import Context
except ImportError:
    try:
        from mcp.server.fastmcp import Context
        from mcp.server.fastmcp import FastMCP as MCPServer
    except ImportError:
        raise ImportError(
            "MCP SDK not installed. Install with: pip install 'mcp[cli]'"
        )

logger = logging.getLogger(__name__)

# ============================================================
# Task Store (in-memory, suitable for stdio single-client mode)
# ============================================================
_task_store: dict[str, TaskInfo] = {}
_completed_results: dict[str, DistillResult] = {}
_distillers: dict[str, Distiller] = {}

# Default output directories (overridable via environment)
DEFAULT_HTML_DIR = os.environ.get(
    "DISTILL_HTML_DIR",
    DEFAULT_HTML_OUTPUT_DIR,
)
DEFAULT_OBSIDIAN_BASE = os.environ.get(
    "DISTILL_OBSIDIAN_DIR",
    DEFAULT_OBSIDIAN_DIR,
)


# ============================================================
# MCP Server
# ============================================================
mcp = MCPServer(
    "doc-distillation-mcp",
    instructions=(
        "A document distillation server supporting Feishu, webpages, PDFs, "
        "and video transcripts. Produces dual output: HTML distillation articles "
        "and Obsidian notes. Uses a five-stage workflow with three-layer image "
        "filtering and integrity verification. For large documents, use "
        "async_mode=True to get a task_id, then poll with get_distill_status."
    ),
)


# ----------------------------------------------------------
# Tool: distill_url
# ----------------------------------------------------------
@mcp.tool()
async def distill_url(
    url: Annotated[str, Field(description="Document URL (Feishu, webpage, PDF, video)")],
    obsidian_subdir: Annotated[
        str | None,
        Field(description="Subdirectory under Obsidian vault for the note (e.g., '飞书蒸馏')"),
    ] = None,
    async_mode: Annotated[
        bool,
        Field(description="If true, return task_id immediately for large documents"),
    ] = False,
    ctx: Context = None,
) -> dict:
    """Distill content from a URL into HTML article + Obsidian note.

    Supports:
    - Feishu documents (feishu.cn, larkoffice.com)
    - Webpages (any HTTP/HTTPS URL)
    - PDF files (.pdf URLs)
    - Video/podcast URLs (YouTube, Bilibili, etc. - returns guidance)

    Output:
    - HTML distillation article saved to ~/Documents/蒸馏文稿/
    - Obsidian note saved to ~/Documents/obsidian/{subdir}/

    For large documents, set async_mode=true and poll with get_distill_status.
    """
    obsidian_dir = DEFAULT_OBSIDIAN_BASE
    if obsidian_subdir:
        obsidian_dir = os.path.join(DEFAULT_OBSIDIAN_BASE, obsidian_subdir)

    distiller = Distiller(
        html_output_dir=DEFAULT_HTML_DIR,
        obsidian_dir=obsidian_dir,
    )

    if async_mode:
        return await _start_async_task(distiller, url, ctx, is_file=False)

    return await _run_sync_distillation(distiller, url, ctx, is_file=False)


# ----------------------------------------------------------
# Tool: distill_file
# ----------------------------------------------------------
@mcp.tool()
async def distill_file(
    file_path: Annotated[str, Field(description="Path to local file (PDF, HTML, TXT, MD, etc.)")],
    obsidian_subdir: Annotated[
        str | None,
        Field(description="Subdirectory under Obsidian vault for the note"),
    ] = None,
    async_mode: Annotated[
        bool,
        Field(description="If true, return task_id for large files"),
    ] = False,
    ctx: Context = None,
) -> dict:
    """Distill content from a local file into HTML article + Obsidian note.

    Supports:
    - PDF files (.pdf)
    - HTML files (.html, .htm)
    - Text/Markdown files (.txt, .md)
    - Audio/Video files (.mp3, .m4a, .wav, .mp4 - returns guidance)

    Output:
    - HTML distillation article saved to ~/Documents/蒸馏文稿/
    - Obsidian note saved to ~/Documents/obsidian/{subdir}/
    """
    obsidian_dir = DEFAULT_OBSIDIAN_BASE
    if obsidian_subdir:
        obsidian_dir = os.path.join(DEFAULT_OBSIDIAN_BASE, obsidian_subdir)

    distiller = Distiller(
        html_output_dir=DEFAULT_HTML_DIR,
        obsidian_dir=obsidian_dir,
    )

    if async_mode:
        return await _start_async_task(distiller, file_path, ctx, is_file=True)

    return await _run_sync_distillation(distiller, file_path, ctx, is_file=True)


# ----------------------------------------------------------
# Tool: get_distill_status
# ----------------------------------------------------------
@mcp.tool()
async def get_distill_status(
    task_id: Annotated[str, Field(description="Task ID from distill_url or distill_file")],
) -> dict:
    """Check the status of an async distillation task.

    Returns the task status, progress, and result (if completed).
    Poll this periodically until status is 'completed' or 'failed'.
    """
    task = _task_store.get(task_id)
    if not task:
        return {"error": f"Task not found: {task_id}"}

    response: dict = {
        "task_id": task.task_id,
        "status": task.status.value,
        "progress": task.progress,
        "message": task.message,
    }

    if task.status == TaskStatus.COMPLETED and task.result:
        response["result"] = task.result.model_dump()
        response["summary"] = _build_summary(task.result)
    elif task.status == TaskStatus.FAILED:
        response["error"] = task.error or "Unknown error"

    return response


# ----------------------------------------------------------
# Tool: list_distillations
# ----------------------------------------------------------
@mcp.tool()
async def list_distillations() -> list[dict]:
    """List all completed distillations.

    Returns a list of completed distillation tasks with metadata.
    Use get_distill_status with a task_id to get full content.
    """
    items: list[dict] = []
    for task_id, task in _task_store.items():
        if task.status != TaskStatus.COMPLETED:
            continue
        result = task.result
        if not result:
            continue
        items.append({
            "task_id": task_id,
            "title": result.doc_info.title,
            "source_type": result.doc_info.source_type.value,
            "method": result.method.value,
            "segment_count": len(result.segments),
            "image_count": len(result.images),
            "distilled_image_count": result.distilled_image_count,
            "html_path": result.html_path,
            "obsidian_path": result.obsidian_path,
            "created_at": task.created_at,
        })

    items.sort(key=lambda x: x["created_at"], reverse=True)
    return items


# ============================================================
# Internal helpers
# ============================================================
async def _run_sync_distillation(
    distiller: Distiller,
    source: str,
    ctx: Context,
    is_file: bool,
) -> dict:
    """Run distillation synchronously with progress reporting."""
    loop = asyncio.get_event_loop()

    def progress_cb(progress: float, total: float | None, message: str) -> None:
        if ctx:
            asyncio.run_coroutine_threadsafe(
                ctx.report_progress(progress, total, message),
                loop,
            )

    def do_distill() -> DistillResult:
        if is_file:
            return distiller.extract_from_file(source, progress_cb=progress_cb)
        else:
            return distiller.extract_from_url(source, progress_cb=progress_cb)

    try:
        result = await loop.run_in_executor(None, do_distill)
    except Exception as e:
        logger.exception("Distillation failed")
        return {"error": f"Distillation failed: {e}"}

    task_id = str(uuid.uuid4())[:8]
    task_info = TaskInfo(
        task_id=task_id,
        status=TaskStatus.COMPLETED,
        progress=1.0,
        message="Completed",
        result=result,
        created_at=time.time(),
        completed_at=time.time(),
    )
    _task_store[task_id] = task_info
    _completed_results[task_id] = result
    _distillers[task_id] = distiller

    response = result.model_dump()
    response["task_id"] = task_id
    response["summary"] = _build_summary(result)
    return response


async def _start_async_task(
    distiller: Distiller,
    source: str,
    ctx: Context,
    is_file: bool,
) -> dict:
    """Start an async distillation task and return task_id immediately."""
    task_id = str(uuid.uuid4())[:8]

    task_info = TaskInfo(
        task_id=task_id,
        status=TaskStatus.PENDING,
        progress=0.0,
        message="Task queued",
        created_at=time.time(),
    )
    _task_store[task_id] = task_info

    asyncio.create_task(
        _run_background_task(task_id, distiller, source, is_file)
    )

    return {
        "task_id": task_id,
        "status": "pending",
        "message": "Distillation started. Use get_distill_status to poll progress.",
    }


async def _run_background_task(
    task_id: str,
    distiller: Distiller,
    source: str,
    is_file: bool,
) -> None:
    """Run distillation in background, updating task store with progress."""
    task = _task_store[task_id]
    task.status = TaskStatus.PROCESSING
    task.message = "Processing..."

    loop = asyncio.get_event_loop()

    def progress_cb(progress: float, total: float | None, message: str) -> None:
        task.progress = progress / total if total else progress
        task.message = message

    def do_distill() -> DistillResult:
        if is_file:
            return distiller.extract_from_file(source, progress_cb=progress_cb)
        else:
            return distiller.extract_from_url(source, progress_cb=progress_cb)

    try:
        result = await loop.run_in_executor(None, do_distill)
        task.status = TaskStatus.COMPLETED
        task.progress = 1.0
        task.message = "Completed"
        task.result = result
        task.completed_at = time.time()
        _completed_results[task_id] = result
        _distillers[task_id] = distiller
        logger.info("Task %s completed: %s", task_id, result.doc_info.title)
    except Exception as e:
        task.status = TaskStatus.FAILED
        task.error = str(e)
        task.message = f"Failed: {e}"
        logger.exception("Task %s failed", task_id)


def _build_summary(result: DistillResult) -> str:
    """Build a human-readable summary of the distillation."""
    lines: list[str] = []
    lines.append(f"Title: {result.doc_info.title}")
    lines.append(f"Author: {result.doc_info.author}")
    lines.append(f"Source type: {result.doc_info.source_type.value}")
    lines.append(f"Method: {result.method.value}")
    lines.append(f"Segments: {len(result.segments)}")
    lines.append(f"Images: {len(result.images)} (distilled: {result.distilled_image_count})")
    if result.structure_skeleton:
        lines.append(f"Structure: {len(result.structure_skeleton)} sections")
    if result.key_elements:
        total_elements = sum(result.key_elements.values())
        lines.append(f"Key elements: {total_elements} detected")
    if result.html_path:
        lines.append(f"HTML: {result.html_path}")
    if result.obsidian_path:
        lines.append(f"Obsidian: {result.obsidian_path}")
    if result.warning:
        lines.append(f"Warning: {result.warning}")
    if result.guidance:
        lines.append(f"Guidance: {result.guidance[:100]}...")
    if result.full_text:
        preview = result.full_text[:200].replace("\n", " ")
        lines.append(f"Preview: {preview}...")
    return "\n".join(lines)


# ============================================================
# Entry point
# ============================================================
def main():
    """Entry point for the MCP server."""
    logging.basicConfig(level=logging.INFO)
    os.makedirs(DEFAULT_HTML_DIR, exist_ok=True)
    os.makedirs(DEFAULT_OBSIDIAN_BASE, exist_ok=True)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
