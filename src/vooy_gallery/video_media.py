"""Expiring, single-video asset links with bounded SQL byte-range reads."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from urllib.parse import urlencode

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import Response

from .media import PhotoMedia

if TYPE_CHECKING:
    from .settings import Settings

logger = logging.getLogger(__name__)
_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_MEDIA_PATH = re.compile(
    r"/video-media/[A-Za-z0-9_-]{1,128}/[A-Za-z0-9_-]{1,128}/(?:original|playback|poster)\Z"
)
_NO_CACHE = {"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff",
             "Referrer-Policy": "no-referrer", "Accept-Ranges": "bytes"}
_MIMES = {"original": {"video/mp4", "video/quicktime"},
          "playback": {"video/mp4"}, "poster": {"image/jpeg"}}
_EXTENSIONS = {"video/mp4": "mp4", "video/quicktime": "mov", "image/jpeg": "jpg"}


def _byte_range(value: str, size: int) -> tuple[int, int]:
    """Parse one inclusive byte range. Multiple/invalid/unsatisfiable ranges fail."""
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
    if match is None or size <= 0:
        raise ValueError("Unsupported byte range")
    first, last = match.groups()
    # Bound integer parsing as well as rejecting an empty range.
    if len(first) > 20 or len(last) > 20 or not (first or last):
        raise ValueError("Invalid byte range")
    if not first:
        suffix = int(last)
        if suffix <= 0:
            raise ValueError("Invalid suffix range")
        return max(0, size - suffix), size - 1
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    if start >= size or end < start:
        raise ValueError("Unsatisfiable byte range")
    return start, end


class VideoMedia(PhotoMedia):
    """Share URL policy with photos, but bind signatures to a separate video domain.

    The HTTP integration must retain Host/Origin checks around ``serve``. A link
    authorizes one whole asset until expiry; a playback time fragment is a player
    hint and does not restrict which parts of that asset the receiver may fetch.
    """

    def __init__(self, settings: Settings, video_store):
        super().__init__(settings, video_store)

    @staticmethod
    def matches_path(path: str) -> bool:
        return _MEDIA_PATH.fullmatch(path) is not None

    def _signature(self, gallery_id: str, video_id: str, variant: str, expires: int) -> str:
        message = json.dumps(
            ["vooy-gallery-video-media-v1", self.base_url, gallery_id, video_id, variant, expires],
            separators=(",", ":"),
        ).encode("utf-8")
        return hmac.new(self._key, message, hashlib.sha256).hexdigest()

    def link(self, video_id: str, variant: str = "playback") -> dict:
        if not _IDENTIFIER.fullmatch(video_id) or variant not in _MIMES:
            raise ValueError("Invalid video media identifier")
        expires = int(time.time()) + self.ttl
        query = urlencode({"expires": expires, "sig": self._signature(self.gallery_id, video_id, variant, expires)})
        return {"url": f"{self.base_url}/video-media/{self.gallery_id}/{video_id}/{variant}?{query}",
                "expires_at": datetime.fromtimestamp(expires, UTC).isoformat(), "expires_in_seconds": self.ttl}

    async def serve(self, request: Request) -> Response:
        if request.method not in {"GET", "HEAD"}:
            return Response("Method not allowed", status_code=405, headers={**_NO_CACHE, "Allow": "GET, HEAD"})
        gallery_id = request.path_params["gallery_id"]
        video_id = request.path_params["video_id"]
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
            expires, now = int(expires_text), int(time.time())
            valid = now < expires <= now + self.ttl and hmac.compare_digest(
                signature, self._signature(gallery_id, video_id, variant, expires)
            )
        if not valid:
            return Response("Invalid or expired video link", status_code=403, headers=_NO_CACHE)
        # HEAD ignores Range; it must not materialize the video BYTEA.
        range_header = request.headers.get("range") if request.method == "GET" else None
        # No validators are advertised. An If-Range condition cannot be confirmed,
        # so return the complete representation instead of a partial response.
        if request.headers.get("if-range") is not None:
            range_header = None
        return await run_in_threadpool(self._asset_response, video_id, variant, request.method, range_header)

    def _asset_response(self, video_id: str, variant: str, method: str, range_header: str | None) -> Response:
        try:
            info = self.store.video_asset_info(video_id, self.gallery_id, variant)
            if info is None:
                return Response("Video asset not found", status_code=404, headers=_NO_CACHE)
            size, mime = info["size"], info["mime_type"]
            if not isinstance(size, int) or isinstance(size, bool) or size < 0 or mime not in _MIMES[variant]:
                raise ValueError("Invalid stored video asset information")
            start = end = None
            status = 200
            headers = {**_NO_CACHE,
                       "Content-Disposition": f'{"attachment" if variant == "original" else "inline"}; '
                                              f'filename="{video_id}.{_EXTENSIONS[mime]}"'}
            if range_header is not None:
                try:
                    start, end = _byte_range(range_header, size)
                except ValueError:
                    return Response("Unsupported or unsatisfiable range", status_code=416,
                                    headers={**_NO_CACHE, "Content-Range": f"bytes */{size}"})
                status = 206
                headers["Content-Range"] = f"bytes {start}-{end}/{size}"
            body_size = size if start is None else end - start + 1
            headers["Content-Length"] = str(body_size)
            if method == "HEAD":
                return Response(b"", status_code=status, media_type=mime, headers=headers)
            data, stored_mime = self.store.get_video_asset(
                video_id, self.gallery_id, variant, start=start, end=end
            )
            if len(data) != body_size or stored_mime != mime:
                raise ValueError("Stored video bytes do not match asset information")
            return Response(data, status_code=status, media_type=mime, headers=headers)
        except (KeyError, ValueError):
            return Response("Video asset unavailable", status_code=404, headers=_NO_CACHE)
        except Exception as exc:  # noqa: BLE001 -- never expose database credentials or paths.
            logger.error("Video media read failed (%s)", type(exc).__name__)
            return Response("Video asset unavailable", status_code=503, headers=_NO_CACHE)
