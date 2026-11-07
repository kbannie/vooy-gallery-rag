"""MCP video search and selection. Video bytes travel through signed HTTP links."""
from __future__ import annotations

import base64
import json
import logging
from typing import TYPE_CHECKING, Any

from mcp.types import CallToolResult, ImageContent, ResourceLink, TextContent, ToolAnnotations

from .media import MAX_THUMBNAIL_BYTES, validated_mime

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from .settings import Settings
    from .video_media import VideoMedia

logger = logging.getLogger(__name__)


def _text(payload: dict) -> TextContent:
    return TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, allow_nan=False))


def _error(code: str, message: str) -> CallToolResult:
    payload = {"status": "error", "code": code, "message": message}
    return CallToolResult(isError=True, content=[_text(payload)], structuredContent=payload)


def _poster(data: bytes) -> ImageContent:
    mime = validated_mime(data, MAX_THUMBNAIL_BYTES)
    if mime != "image/jpeg":
        raise ValueError("A video poster must be JPEG")
    return ImageContent(type="image", data=base64.b64encode(data).decode("ascii"), mimeType=mime)


def _bounds(segment: dict, video: dict, gallery_id: str) -> tuple[int, int]:
    if (segment.get("video_id") != video["video_id"]
            or segment.get("gallery_id", gallery_id) != gallery_id):
        raise ValueError("Segment belongs to a different video or gallery")
    start, end, duration = segment.get("start_ms"), segment.get("end_ms"), video.get("duration_ms")
    if (any(not isinstance(x, int) or isinstance(x, bool) for x in (start, end, duration))
            or not 0 <= start < end <= duration):
        raise ValueError("Invalid segment bounds")
    return start, end


def _seconds(milliseconds: int) -> str:
    return f"{milliseconds // 1000}.{milliseconds % 1000:03d}".rstrip("0").rstrip(".")


def register_video_tools(
    server: FastMCP, settings: Settings, video_store, media: VideoMedia | None, search_service: Any | None = None
) -> None:
    """Register video tools only when the parent server enables video support.

    Media HTTP routes and Bearer/Host/Origin checks are installed by the parent.
    A playback fragment is a start/end hint for a full-video player, not a clipped
    file or an authorization restriction. No tool sends a chat message.
    """

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=True))
    def search_videos(query: str, gallery_id: str | None = None, limit: int = 3) -> CallToolResult:
        """Find up to three video segments and show their frame previews for user selection.

        Results identify both video_id and segment_id. Call get_video with both IDs
        after the user chooses. Inline previews come from the matched segment;
        poster_url, when present, is the whole video's cover and may differ.
        Do not append unsolicited source, account or license commentary.
        """
        nonlocal search_service
        gallery_id = gallery_id or settings.gallery_id
        if gallery_id != settings.gallery_id:
            return _error("gallery_not_available", "This gallery is not available in this local session.")
        if not query.strip() or len(query) > 4000:
            return _error("invalid_query", "query must contain 1–4000 characters.")
        if not 1 <= limit <= 3:
            return _error("invalid_limit", "limit must be between 1 and 3.")
        try:
            if search_service is None:
                from .video_search import VideoSearchService

                search_service = VideoSearchService(video_store, settings)
            raw = search_service.search(query, gallery_id=gallery_id, limit=limit)
        except Exception as exc:  # noqa: BLE001 -- provider failures must not reveal credentials.
            logger.error("Video MCP search failed (%s)", type(exc).__name__)
            return _error("search_failed", "Video search could not run. Check configuration and indexing status.")
        public = {key: raw[key] for key in (
            "result_id", "status", "query", "gallery_id", "clarification", "applied_filters",
            "evidence_scope", "evidence_note"
        ) if key in raw}
        public["results"] = []
        content: list[TextContent | ImageContent] = [TextContent(type="text", text="")]
        for item in raw.get("results", [])[:limit]:
            try:
                video = video_store.get_video(item["video_id"], gallery_id)
                segment = video_store.get_segment(item["segment_id"], gallery_id)
                if video is None or segment is None:
                    return _error("invalid_search_result", "A search result is no longer available in this gallery.")
                start, end = _bounds(segment, video, gallery_id)
            except ValueError:
                return _error("invalid_search_result", "A search segment does not belong to its video.")
            except Exception as exc:  # noqa: BLE001 -- sanitize metadata/storage failures.
                logger.error("Video MCP lookup failed (%s)", type(exc).__name__)
                return _error("lookup_failed", "Video metadata could not be read from the gallery database.")
            result = {key: item[key] for key in ("caption", "reason", "score", "evidence_scope") if key in item}
            result.update(video_id=video["video_id"], segment_id=segment["segment_id"], start_ms=start, end_ms=end,
                          metadata=video.get("metadata", {}))
            try:
                image = _poster(video_store.get_segment_poster(segment["segment_id"], gallery_id))
                content.append(TextContent(type="text", text=f"Video {len(public['results']) + 1}: {video['video_id']}"))
                result.update(preview_content_index=len(content), preview_kind="matched_segment_frame")
                content.append(image)
            except Exception as exc:  # noqa: BLE001 -- invalid/missing frame preview does not leak details.
                logger.warning("Video MCP preview unavailable (%s)", type(exc).__name__)
                result["preview_unavailable"] = True
            if media is not None:
                link = media.link(video["video_id"], "poster")
                result.update(poster_url=link["url"], poster_url_expires_at=link["expires_at"],
                              poster_kind="whole_video_cover")
            public["results"].append(result)
        content[0] = _text(public)
        return CallToolResult(content=content, structuredContent=public)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
    def get_video(video_id: str, gallery_id: str | None = None, segment_id: str | None = None) -> CallToolResult:
        """Return signed playback/download links for a selected video, without sending it.

        Supply the search result's segment_id to begin playback at that segment.
        The fragment is a player hint; the URL still grants access to the full
        video until expiry. The original download always contains the full video.
        Use the host's attachment tool or show the link; do not claim delivery.
        Do not append unsolicited source, account or license commentary.
        MP4 bytes are never embedded as a base64 MCP content block.
        """
        gallery_id = gallery_id or settings.gallery_id
        if gallery_id != settings.gallery_id:
            return _error("gallery_not_available", "This gallery is not available in this local session.")
        if media is None:
            return _error("media_url_unavailable", "Configure an authenticated public media URL before opening videos.")
        try:
            video = video_store.get_video(video_id, gallery_id)
            if video is None:
                return _error("video_not_found", "The requested video does not exist in this gallery.")
            selected_segment = None
            if segment_id is not None:
                selected_segment = video_store.get_segment(segment_id, gallery_id)
                if selected_segment is None:
                    return _error("segment_not_found", "The requested segment does not exist in this gallery.")
                try:
                    start, end = _bounds(selected_segment, video, gallery_id)
                except ValueError:
                    return _error("segment_not_in_video", "The segment does not belong to this video or has invalid bounds.")
            playback_info = video_store.video_asset_info(video_id, gallery_id, "playback")
            original_info = video_store.video_asset_info(video_id, gallery_id, "original")
            if playback_info is None or original_info is None:
                return _error("video_unavailable", "A playable or original video asset is unavailable.")
            if playback_info["mime_type"] != "video/mp4":
                return _error("video_unavailable", "The playback asset is not a supported MP4 video.")
            playback = media.link(video_id, "playback")
            original = media.link(video_id, "original")
            poster_link = media.link(video_id, "poster")
        except Exception as exc:  # noqa: BLE001 -- external MCP boundary hides storage details.
            logger.error("Video MCP lookup failed (%s)", type(exc).__name__)
            return _error("lookup_failed", "Video metadata or media links could not be read.")
        playback_url = playback["url"]
        payload = {"status": "ok", "video_id": video_id, "gallery_id": gallery_id,
                   "duration_ms": video["duration_ms"], "metadata": video.get("metadata", {}),
                   "mime_type": "video/mp4", "byte_size": playback_info["size"],
                   "filename": f"{video_id}.mp4", "video_url": playback_url,
                   "playback_url": playback_url, "video_url_expires_at": playback["expires_at"],
                   "video_url_expires_in_seconds": playback["expires_in_seconds"],
                   "original_download_url": original["url"],
                   "original_download_url_expires_at": original["expires_at"],
                   "original_mime_type": original_info["mime_type"], "original_byte_size": original_info["size"],
                   "poster_url": poster_link["url"], "poster_url_expires_at": poster_link["expires_at"],
                   "poster_kind": "whole_video_cover", "delivery_status": "not_sent"}
        if selected_segment is not None:
            playback_url += f"#t={_seconds(start)},{_seconds(end)}"
            payload.update(segment_id=segment_id, start_ms=start, end_ms=end, playback_url=playback_url,
                           playback_scope="full_video_with_segment_time_hint")
        content: list[TextContent | ResourceLink | ImageContent] = [
            _text(payload), ResourceLink(type="resource_link", uri=playback_url, name=payload["filename"],
                                        mimeType="video/mp4", size=playback_info["size"],
                                        description=f"Full video playback; expires {playback['expires_at']}.")]
        try:
            if selected_segment is not None:
                image = _poster(video_store.get_segment_poster(segment_id, gallery_id))
            else:
                poster_bytes, _ = video_store.get_video_asset(video_id, gallery_id, "poster")
                image = _poster(poster_bytes)
            content.append(image)
        except Exception as exc:  # noqa: BLE001 -- poster failure must not prevent a valid playback URL.
            logger.warning("Video MCP poster unavailable (%s)", type(exc).__name__)
        return CallToolResult(content=content, structuredContent=payload)
