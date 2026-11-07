"""Local, read-only MCP boundary for photo retrieval.

Vooy registration and message delivery are deliberately outside this server:
MCP image results are not evidence that a messenger posted an attachment.
"""
from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import logging
import secrets
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from mcp.server.auth.middleware.auth_context import AuthContextMiddleware
from mcp.server.auth.middleware.bearer_auth import BearerAuthBackend, RequireAuthMiddleware
from mcp.server.auth.provider import AccessToken
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecurityMiddleware
from mcp.types import CallToolResult, ImageContent, ResourceLink, TextContent, ToolAnnotations
from starlette.applications import Starlette
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send

from .media import IMAGE_EXTENSIONS, PhotoMedia, validated_mime

if TYPE_CHECKING:
    from .settings import Settings
    from .store import Store

logger = logging.getLogger(__name__)
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_THUMBNAIL_BYTES = 2 * 1024 * 1024


class _StaticTokenVerifier:
    """Verify a dedicated, pre-shared MCP token without claiming an OAuth issuer."""

    def __init__(self, token: str):
        if not token.strip():
            raise ValueError("MCP access token must not be blank")
        self._expected = token.encode("utf-8")

    async def verify_token(self, token: str) -> AccessToken | None:
        if not secrets.compare_digest(token.encode("utf-8"), self._expected):
            return None
        return AccessToken(token=token, client_id="vooy-gallery", scopes=["photos:read"])


class _HTTPBearerMiddleware:
    """Protect HTTP requests while preserving the SDK app's lifespan handling."""

    def __init__(self, app: ASGIApp, verifier: _StaticTokenVerifier, media: PhotoMedia | None = None,
                 video_media: PhotoMedia | None = None):
        self.app = app
        self.media = media
        self.video_media = video_media
        self.authenticated_app = AuthenticationMiddleware(
            AuthContextMiddleware(RequireAuthMiddleware(app, required_scopes=["photos:read"])),
            backend=BearerAuthBackend(verifier),
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (scope["type"] == "http" and scope["method"] in {"GET", "HEAD"}
                and any(media is not None and media.matches_path(scope["path"])
                        for media in (self.media, self.video_media))):
            # The media endpoint authenticates the expiring single-photo signature itself.
            await self.app(scope, receive, send)
            return
        target = self.authenticated_app if scope["type"] == "http" else self.app
        await target(scope, receive, send)


class _GalleryFastMCP(FastMCP):
    def __init__(self, *args: Any, access_token: str | None = None, media: PhotoMedia | None = None,
                 video_media: PhotoMedia | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._gallery_verifier = _StaticTokenVerifier(access_token) if access_token is not None else None
        self._gallery_media = media
        self._video_media = video_media

    def streamable_http_app(self) -> Starlette:
        # FastMCP.run() calls this method too: CLI serving must use the same auth boundary.
        app = super().streamable_http_app()
        if self._gallery_verifier is not None:
            app.add_middleware(_HTTPBearerMiddleware, verifier=self._gallery_verifier,
                               media=self._gallery_media, video_media=self._video_media)
        return app

    async def run_streamable_http_async(self) -> None:
        # Signed URLs are credentials for individual images. Do not put query strings in access logs.
        import uvicorn

        config = uvicorn.Config(self.streamable_http_app(), host=self.settings.host, port=self.settings.port,
                                log_level=self.settings.log_level.lower(), access_log=False)
        await uvicorn.Server(config).serve()


def _local_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _image_content(data: bytes, maximum: int) -> ImageContent:
    """Validate bounded image bytes read from the authoritative PostgreSQL asset."""
    mime_type = validated_mime(data, maximum)
    return ImageContent(type="image", data=base64.b64encode(data).decode("ascii"), mimeType=mime_type)


def _error(code: str, message: str) -> CallToolResult:
    payload = {"status": "error", "code": code, "message": message}
    return CallToolResult(
        isError=True,
        content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
        structuredContent=payload,
    )


def _text(payload: dict) -> TextContent:
    return TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, allow_nan=False))


def build_server(
    settings: Settings, store: Store | None = None, search_service: Any | None = None,
    video_store: Any | None = None, video_search_service: Any | None = None,
) -> FastMCP:
    """Build a loopback FastMCP server; injectable services keep protocol tests offline.

    Start with ``build_server(settings).run(transport="streamable-http")``.
    The application pins the official MCP Python SDK to its 1.x line.
    """
    if not _local_host(settings.mcp_host):
        raise ValueError("This local-only version requires a loopback MCP_HOST")
    if store is None:
        from .store import Store

        store = Store(settings.data_dir, database_url=settings.database_url)

    media = PhotoMedia(settings, store) if settings.mcp_public_base_url else None
    video_media = None
    if settings.video_enabled:
        from .video_media import VideoMedia
        from .video_store import VideoStore

        if settings.video_gallery_id == settings.gallery_id:
            raise ValueError("VIDEO_GALLERY_ID must differ from the photo GALLERY_ID")
        video_settings = replace(settings, gallery_id=settings.video_gallery_id)
        video_store = video_store if video_store is not None else VideoStore(store)
        video_media = VideoMedia(video_settings, video_store) if settings.mcp_public_base_url else None
    server = _GalleryFastMCP(
        "BTS Gallery",
        access_token=settings.mcp_access_token,
        media=media,
        video_media=video_media,
        instructions=(
            "Search the configured photo gallery, show up to three numbered candidates, "
            "and ask the user to select a photo before calling get_photo. "
            "Synthetic place/time metadata describes a demo fixture, not actual BTS events. "
            "Keep chat replies focused on the requested media and search details. "
            "Do not append unsolicited source, author/account, license or usage-permission commentary. "
            "This server does not send messages. "
            "When get_photo returns image_url, use that direct image URL for the host's image attachment "
            "capability. If attachment is unavailable, show the image_url as a clickable photo link. "
            "Never claim delivery or a support report "
            "succeeded without a successful corresponding host tool result."
            " When video tools are available, use search_videos for videos and get_video with both "
            "the chosen video_id and segment_id. Video playback links expire; request a fresh link when needed."
        ),
        host=settings.mcp_host,
        port=settings.mcp_port,
        streamable_http_path="/mcp",
        json_response=True,
        stateless_http=True,
    )
    if media is not None:
        security = TransportSecurityMiddleware(server.settings.transport_security)

        @server.custom_route("/media/{gallery_id}/{photo_id}/{variant}", methods=["GET", "HEAD"])
        async def photo_media(request: Request) -> Response:
            rejected = await security.validate_request(request)
            return rejected if rejected is not None else await media.serve(request)
    if settings.video_enabled:
        from .video_tools import register_video_tools

        register_video_tools(server, video_settings, video_store, video_media, video_search_service)
        if video_media is not None:
            video_security = TransportSecurityMiddleware(server.settings.transport_security)

            @server.custom_route("/video-media/{gallery_id}/{video_id}/{variant}", methods=["GET", "HEAD"])
            async def video_asset(request: Request) -> Response:
                rejected = await video_security.validate_request(request)
                return rejected if rejected is not None else await video_media.serve(request)

    def allowed_gallery(gallery_id: str) -> bool:
        return gallery_id == settings.gallery_id

    @server.tool(
        annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=True)
    )
    def search_photos(query: str, gallery_id: str | None = None, limit: int = 3) -> CallToolResult:
        """Find up to three candidate photos. Show previews and let the user choose.

        Returns ok, no_match, or needs_clarification with photo IDs and search details.
        Preview images appear in content blocks referenced by preview_content_index.
        If available, preview_url opens the same database thumbnail without a bearer
        header until preview_url_expires_at. Do not append source or license commentary.
        This tool neither selects a photo nor shares one to a conversation.
        """
        nonlocal search_service
        gallery_id = gallery_id or settings.gallery_id
        if not allowed_gallery(gallery_id):
            return _error("gallery_not_available", "This gallery is not available in this local session.")
        if not query.strip() or len(query) > 4000:
            return _error("invalid_query", "query must contain 1–4000 characters.")
        if not 1 <= limit <= 3:
            return _error("invalid_limit", "limit must be between 1 and 3.")
        try:
            if search_service is None:
                from .retrieval import SearchService

                search_service = SearchService(store, settings)
            raw = search_service.search(query, gallery_id=gallery_id, limit=limit)
        except Exception as exc:  # noqa: BLE001 -- sanitize provider failures at the external boundary.
            # Do not return provider errors, database paths, or credentials to MCP hosts.
            logger.error("Gallery MCP search failed (%s)", type(exc).__name__)
            return _error(
                "search_failed",
                "Search could not run. Check OpenAI configuration and indexing status in the local CLI.",
            )

        public = {
            key: raw[key]
            for key in ("result_id", "status", "query", "gallery_id", "clarification", "applied_filters")
            if key in raw
        }
        public["results"] = []
        content: list[TextContent | ImageContent] = [TextContent(type="text", text="")]
        for item in raw.get("results", [])[:limit]:
            try:
                photo = store.get_photo(item["photo_id"], gallery_id)
            except Exception as exc:  # noqa: BLE001 -- external boundary never discloses DB connection details.
                logger.error("Gallery MCP lookup failed (%s)", type(exc).__name__)
                return _error("lookup_failed", "Photo metadata could not be read from the gallery database.")
            if photo is None:
                return _error("invalid_search_result", "Search returned a photo outside this gallery.")
            public_item = {
                key: item[key]
                for key in ("photo_id", "caption", "score", "reason")
                if key in item
            }
            public_item.update(metadata=photo.get("metadata", {}))
            try:
                preview_bytes, _ = store.get_asset(photo["photo_id"], gallery_id, thumbnail=True)
                preview = _image_content(preview_bytes, MAX_THUMBNAIL_BYTES)
                content.append(TextContent(type="text", text=f"Photo {len(public['results']) + 1}: {photo['photo_id']}"))
                public_item["preview_content_index"] = len(content)
                content.append(preview)
                if media is not None:
                    link = media.link(photo["photo_id"], "thumbnail")
                    public_item.update(preview_url=link["url"], preview_url_expires_at=link["expires_at"])
            except Exception as exc:  # noqa: BLE001 -- absent or invalid DB thumbnails do not expose details.
                logger.warning("Gallery MCP thumbnail unavailable (%s)", type(exc).__name__)
                public_item["preview_unavailable"] = True
            public["results"].append(public_item)
        content[0] = _text(public)
        return CallToolResult(content=content, structuredContent=public)

    @server.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
        )
    )
    def get_photo(photo_id: str, gallery_id: str | None = None) -> CallToolResult:
        """Return the selected original image, without sending any message.

        Only the configured gallery is accessible. Ask the user to select a search
        candidate first. Original images are limited to 20 MiB per MCP response.
        A host must implement its own conversation attachment and delivery behavior.
        When present, image_url is a temporary direct image URL usable without a
        bearer header; use it for attachments or show it as a clickable photo link.
        Do not append source, account or license commentary. Fetch get_photo
        again for a fresh URL after image_url_expires_at. Never claim a message was sent.
        """
        gallery_id = gallery_id or settings.gallery_id
        if not allowed_gallery(gallery_id):
            return _error("gallery_not_available", "This gallery is not available in this local session.")
        try:
            photo = store.get_photo(photo_id, gallery_id)
        except Exception as exc:  # noqa: BLE001 -- sanitize database failures at the MCP boundary.
            logger.error("Gallery MCP lookup failed (%s)", type(exc).__name__)
            return _error("lookup_failed", "Photo metadata could not be read from the gallery database.")
        if photo is None:
            return _error("photo_not_found", "The requested photo does not exist in this gallery.")
        try:
            image_bytes, _ = store.get_asset(photo_id, gallery_id)
            image = _image_content(image_bytes, MAX_IMAGE_BYTES)
        except Exception as exc:  # noqa: BLE001 -- no database path/credentials in a tool result.
            logger.warning("Gallery MCP original unavailable (%s)", type(exc).__name__)
            return _error("image_unavailable", "The image is missing, damaged, or exceeds the 20 MiB MCP limit.")
        payload = {
            "status": "ok",
            "photo_id": photo_id,
            "gallery_id": gallery_id,
            "mime_type": image.mimeType,
            "filename": f"{photo_id}.{IMAGE_EXTENSIONS[image.mimeType]}",
            "byte_size": len(image_bytes),
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "image_content_index": 1,
            "metadata": photo.get("metadata", {}),
            "delivery_status": "not_sent",
        }
        links: list[ResourceLink] = []
        if media is not None:
            link = media.link(photo_id)
            payload.update(image_url=link["url"], image_url_expires_at=link["expires_at"],
                           image_url_expires_in_seconds=link["expires_in_seconds"],
                           display_hint="Use image_url for a host image attachment, or show it as a clickable photo link.")
            links.append(ResourceLink(type="resource_link", uri=link["url"], name=payload["filename"],
                                      mimeType=image.mimeType, size=len(image_bytes),
                                      description=f"Selected original photo; expires {link['expires_at']}."))
        return CallToolResult(content=[_text(payload), image, *links], structuredContent=payload)

    return server
