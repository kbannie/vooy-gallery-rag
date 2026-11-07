"""Short-lived, single-asset HTTP links for hosts that cannot attach MCP image blocks."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import time
from datetime import UTC, datetime
from io import BytesIO
from typing import TYPE_CHECKING
from urllib.parse import urlencode, urlsplit

from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import Response

if TYPE_CHECKING:
    from .settings import Settings
    from .store import Store

logger = logging.getLogger(__name__)
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_THUMBNAIL_BYTES = 2 * 1024 * 1024
IMAGE_MIMES = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
IMAGE_EXTENSIONS = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}
_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_MEDIA_PATH = re.compile(r"/media/[A-Za-z0-9_-]{1,128}/[A-Za-z0-9_-]{1,128}/(?:original|thumbnail)\Z")
_NO_CACHE = {"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff",
             "Referrer-Policy": "no-referrer"}


def validated_mime(data: bytes, maximum: int) -> str:
    if len(data) > maximum:
        raise ValueError("Image exceeds the image size limit")
    try:
        with Image.open(BytesIO(data)) as picture:
            mime = IMAGE_MIMES.get(picture.format)
            if mime is None:
                raise ValueError("Unsupported gallery image format")
            picture.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Gallery image is damaged or unsupported") from exc
    return mime


class PhotoMedia:
    """A link grants access only to its gallery/photo/variant until expiry.

    The MCP bearer token never appears in a URL. Signatures use domain-separated
    HMAC; links do not authorize MCP requests or enumerate other photos.
    """

    def __init__(self, settings: Settings, store: Store):
        base = (settings.mcp_public_base_url or "").rstrip("/")
        parsed = urlsplit(base)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.path or parsed.query or parsed.fragment
                or any(char.isspace() for char in base)):
            raise ValueError("MCP_PUBLIC_BASE_URL must be an HTTPS origin without a path, credentials or query")
        if not settings.mcp_access_token or not settings.mcp_access_token.strip():
            raise ValueError("MCP_PUBLIC_BASE_URL requires MCP_ACCESS_TOKEN")
        if not 60 <= settings.mcp_image_url_ttl_seconds <= 3600:
            raise ValueError("MCP_IMAGE_URL_TTL_SECONDS must be between 60 and 3600")
        if not _IDENTIFIER.fullmatch(settings.gallery_id):
            raise ValueError("The gallery ID is not a valid media identifier")
        self.base_url = base
        self.gallery_id = settings.gallery_id
        self.ttl = settings.mcp_image_url_ttl_seconds
        self._key = settings.mcp_access_token.encode("utf-8")
        self.store = store

    @staticmethod
    def matches_path(path: str) -> bool:
        return _MEDIA_PATH.fullmatch(path) is not None

    def _signature(self, gallery_id: str, photo_id: str, variant: str, expires: int) -> str:
        message = json.dumps(["vooy-gallery-media-v1", self.base_url, gallery_id, photo_id, variant, expires],
                             separators=(",", ":")).encode("utf-8")
        return hmac.new(self._key, message, hashlib.sha256).hexdigest()

    def link(self, photo_id: str, variant: str = "original") -> dict:
        if not _IDENTIFIER.fullmatch(photo_id) or variant not in {"original", "thumbnail"}:
            raise ValueError("Invalid media identifier")
        expires = int(time.time()) + self.ttl
        query = urlencode({"expires": expires, "sig": self._signature(self.gallery_id, photo_id, variant, expires)})
        return {
            "url": f"{self.base_url}/media/{self.gallery_id}/{photo_id}/{variant}?{query}",
            "expires_at": datetime.fromtimestamp(expires, UTC).isoformat(),
            "expires_in_seconds": self.ttl,
        }

    async def serve(self, request: Request) -> Response:
        gallery_id = request.path_params["gallery_id"]
        photo_id = request.path_params["photo_id"]
        variant = request.path_params["variant"]
        pairs = request.query_params.multi_items()
        expires_text = request.query_params.get("expires", "")
        signature = request.query_params.get("sig", "")
        valid = (
            self.matches_path(request.url.path) and gallery_id == self.gallery_id
            and len(pairs) == 2 and {key for key, _ in pairs} == {"expires", "sig"}
            and re.fullmatch(r"[0-9]{1,12}", expires_text) is not None
            and re.fullmatch(r"[0-9a-f]{64}", signature) is not None
        )
        if valid:
            expires = int(expires_text)
            now = int(time.time())
            valid = now < expires <= now + self.ttl and hmac.compare_digest(
                signature, self._signature(gallery_id, photo_id, variant, expires)
            )
        if not valid:
            return Response("Invalid or expired image link", status_code=403, headers=_NO_CACHE)
        return await run_in_threadpool(self._asset_response, photo_id, variant, request.method)

    def _asset_response(self, photo_id: str, variant: str, method: str) -> Response:
        try:
            if self.store.get_photo(photo_id, self.gallery_id) is None:
                return Response("Image not found", status_code=404, headers=_NO_CACHE)
            thumbnail = variant == "thumbnail"
            data, _ = self.store.get_asset(photo_id, self.gallery_id, thumbnail=thumbnail)
            mime = validated_mime(data, MAX_THUMBNAIL_BYTES if thumbnail else MAX_IMAGE_BYTES)
        except ValueError:
            return Response("Image unavailable", status_code=404, headers=_NO_CACHE)
        except Exception as exc:  # noqa: BLE001 -- never expose storage errors or credentials.
            logger.error("Gallery media read failed (%s)", type(exc).__name__)
            return Response("Image unavailable", status_code=503, headers=_NO_CACHE)
        headers = {
            **_NO_CACHE, "Content-Length": str(len(data)),
            "Content-Disposition": f'inline; filename="{photo_id}.{IMAGE_EXTENSIONS[mime]}"',
        }
        return Response(b"" if method == "HEAD" else data, media_type=mime, headers=headers)
