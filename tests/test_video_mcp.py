import base64
from copy import deepcopy
from dataclasses import replace
from io import BytesIO
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from PIL import Image
from starlette.applications import Starlette
from starlette.routing import Route

from vooy_gallery.mcp_server import build_server
from vooy_gallery.media import PhotoMedia
from vooy_gallery.settings import Settings
from vooy_gallery.video_media import VideoMedia
from vooy_gallery.video_tools import register_video_tools


class VideoStoreDouble:
    def __init__(self, gallery_id):
        image = BytesIO()
        Image.new("RGB", (24, 16), "blue").save(image, format="JPEG")
        self.poster = image.getvalue()
        self.videos = {
            "video-1": {"video_id": "video-1", "gallery_id": gallery_id, "duration_ms": 9000,
                        "title": "private-original-name.mov", "original_path": "/private/personal-video.mov",
                        "metadata": {"origin": "exif", "location": None, "captured_at": None},
                        "source": {"attribution": "사용자 제공 영상", "page_url": "", "license_url": "",
                                   "local_path": "/private/personal-video.mov"}},
            "video-2": {"video_id": "video-2", "gallery_id": gallery_id, "duration_ms": 9000,
                        "metadata": {}, "source": {}},
        }
        self.segments = {
            "segment-1": {"segment_id": "segment-1", "video_id": "video-1", "gallery_id": gallery_id,
                          "start_ms": 1250, "end_ms": 3500},
            "segment-2": {"segment_id": "segment-2", "video_id": "video-2", "gallery_id": gallery_id,
                          "start_ms": 4000, "end_ms": 6000},
        }
        self.assets = {
            "original": (b"original-mov-data-is-distinct", "video/quicktime"),
            "playback": (b"0123456789abcdefghij", "video/mp4"),
            "poster": (self.poster, "image/jpeg"),
        }
        self.accesses = []

    def get_video(self, video_id, gallery_id):
        self.accesses.append(("video", video_id, gallery_id))
        video = self.videos.get(video_id)
        return deepcopy(video) if video and video["gallery_id"] == gallery_id else None

    def get_segment(self, segment_id, gallery_id):
        self.accesses.append(("segment", segment_id, gallery_id))
        segment = self.segments.get(segment_id)
        return deepcopy(segment) if segment and segment["gallery_id"] == gallery_id else None

    def get_segment_poster(self, segment_id, gallery_id):
        if not self.get_segment(segment_id, gallery_id):
            raise ValueError("Missing segment")
        return self.poster

    def video_asset_info(self, video_id, gallery_id, variant):
        self.accesses.append(("info", video_id, gallery_id, variant))
        if not self.get_video(video_id, gallery_id):
            return None
        data, mime = self.assets[variant]
        return {"size": len(data), "mime_type": mime}

    def get_video_asset(self, video_id, gallery_id, variant, start=None, end=None):
        self.accesses.append(("bytes", video_id, gallery_id, variant, start, end))
        data, mime = self.assets[variant]
        if start is not None:
            data = data[start:end + 1]
        return data, mime


class VideoSearchDouble:
    def __init__(self, status="ok", segment_id="segment-1"):
        self.status = status
        self.segment_id = segment_id
        self.calls = []

    def search(self, query, gallery_id=None, limit=3):
        self.calls.append((query, gallery_id, limit))
        return {"result_id": "video-result-1", "status": self.status, "query": query,
                "gallery_id": gallery_id, "applied_filters": {},
                "evidence_scope": "sampled_frames", "evidence_note": "샘플 프레임 관찰 결과입니다.",
                "clarification": "어느 날짜인가요?" if self.status == "needs_clarification" else None,
                "results": [{"video_id": "video-1", "segment_id": self.segment_id,
                             "start_ms": 0, "end_ms": 1, "caption": {"caption_ko": "정원을 걷는 사람"},
                             "reason": "걷는 동작이 보임", "score": 0.9, "evidence_scope": "sampled_frames",
                             "video_title": "private-original-name.mov", "frame_path": "/private/frame.jpg"}]
                if self.status == "ok" else []}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def video_gallery():
    settings = Settings(gallery_id="local-videos", mcp_access_token="synthetic-video-token",
                        mcp_public_base_url="https://video.example.com", mcp_image_url_ttl_seconds=900)
    store = VideoStoreDouble(settings.gallery_id)
    return settings, store, VideoMedia(settings, store)


def media_app(media):
    return Starlette(routes=[Route("/video-media/{gallery_id}/{video_id}/{variant}",
                                   media.serve, methods=["GET", "HEAD"])])


def video_server(settings, store, media, search=None):
    server = FastMCP("Local video test", host="127.0.0.1", port=8765, stateless_http=True, json_response=True)
    register_video_tools(server, settings, store, media, search or VideoSearchDouble())
    if media is not None:
        server.custom_route("/video-media/{gallery_id}/{video_id}/{variant}", methods=["GET", "HEAD"])(media.serve)
    return server


@pytest.mark.anyio
async def test_official_mcp_client_search_select_and_playback_range(video_gallery):
    settings, store, media = video_gallery
    search = VideoSearchDouble()
    server = video_server(settings, store, media, search)
    app = server.streamable_http_app()
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as http_client,
        streamable_http_client("http://localhost:8765/mcp", http_client=http_client) as streams,
        ClientSession(streams[0], streams[1]) as client,
    ):
        await client.initialize()
        tools = await client.list_tools()
        assert {t.name for t in tools.tools} == {"search_videos", "get_video"}
        found = await client.call_tool("search_videos", {"query": "정원을 걷는 영상"})
        assert not found.isError
        candidate = found.structuredContent["results"][0]
        assert candidate["video_id"] == "video-1"
        assert candidate["segment_id"] == "segment-1"
        assert (candidate["start_ms"], candidate["end_ms"]) == (1250, 3500)
        assert candidate["preview_kind"] == "matched_segment_frame"
        assert candidate["poster_kind"] == "whole_video_cover"
        image = found.content[candidate["preview_content_index"]]
        assert image.mimeType == "image/jpeg"
        assert base64.b64decode(image.data) == store.poster
        assert found.structuredContent["result_id"] == "video-result-1"
        assert found.structuredContent["evidence_scope"] == "sampled_frames"
        assert found.structuredContent["evidence_note"]
        assert candidate["evidence_scope"] == "sampled_frames"
        selected = await client.call_tool("get_video", {
            "video_id": candidate["video_id"], "segment_id": candidate["segment_id"],
        })
        assert not selected.isError
        payload = selected.structuredContent
        assert payload["gallery_id"] == "local-videos"
        assert payload["playback_url"].endswith("#t=1.25,3.5")
        assert urlsplit(payload["video_url"]).fragment == ""
        assert "/playback?" in payload["video_url"]
        assert "/original?" in payload["original_download_url"]
        assert payload["original_mime_type"] == "video/quicktime"
        assert payload["byte_size"] != payload["original_byte_size"]
        assert payload["delivery_status"] == "not_sent"
        assert payload["playback_scope"] == "full_video_with_segment_time_hint"
        assert selected.content[1].type == "resource_link"
        assert selected.content[1].mimeType == "video/mp4"
        assert str(selected.content[1].uri) == payload["playback_url"]
        for result in [found, selected]:
            text = result.model_dump_json()
            assert settings.mcp_access_token not in text
            assert "private-original-name" not in text
            assert "/private/" not in text
            assert all(c.type != "image" or c.mimeType == "image/jpeg" for c in result.content)
        assert not any(a[0] == "bytes" and a[3] in {"original", "playback"} for a in store.accesses)
        response = await http_client.get(payload["playback_url"], headers={"Range": "bytes=2-5"})
        assert response.status_code == 206
        assert response.content == b"2345"
        assert response.headers["content-range"] == "bytes 2-5/20"
        original = await http_client.get(payload["original_download_url"])
        assert original.status_code == 200
        assert original.content == store.assets["original"][0]
        assert original.headers["content-type"] == "video/quicktime"
        assert original.headers["content-disposition"].startswith("attachment;")
    assert search.calls == [("정원을 걷는 영상", "local-videos", 3)]


@pytest.mark.anyio
@pytest.mark.parametrize("variant,mime", [
    ("original", "video/quicktime"), ("playback", "video/mp4"), ("poster", "image/jpeg"),
])
async def test_signed_video_get_and_head_preserve_asset_type_without_head_blob_read(video_gallery, variant, mime):
    _, store, media = video_gallery
    url = media.link("video-1", variant)["url"]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=media_app(media))) as client:
        response = await client.get(url)
        assert response.status_code == 200
        assert response.headers["content-type"] == mime
        assert response.content == store.assets[variant][0]
        assert "no-store" in response.headers["cache-control"]
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "no-referrer"
        store.accesses.clear()
        head = await client.head(url, headers={"Range": "bytes=0-1"})
        assert head.status_code == 200
        assert head.content == b""
        assert int(head.headers["content-length"]) == len(store.assets[variant][0])
        assert not any(a[0] == "bytes" for a in store.accesses)


@pytest.mark.anyio
@pytest.mark.parametrize("range_header,start,end", [
    ("bytes=0-0", 0, 0), ("bytes=4-7", 4, 7), ("bytes=15-", 15, 19),
    ("bytes=-3", 17, 19), ("bytes=-999", 0, 19), ("bytes=18-999", 18, 19),
])
async def test_range_reads_only_requested_sql_slice(video_gallery, range_header, start, end):
    _, store, media = video_gallery
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=media_app(media))) as client:
        response = await client.get(media.link("video-1")["url"], headers={"Range": range_header})
    assert response.status_code == 206
    assert response.content == store.assets["playback"][0][start:end + 1]
    assert response.headers["content-range"] == f"bytes {start}-{end}/20"
    assert int(response.headers["content-length"]) == end - start + 1
    assert [a for a in store.accesses if a[0] == "bytes"] == [
        ("bytes", "video-1", "local-videos", "playback", start, end)
    ]


@pytest.mark.anyio
@pytest.mark.parametrize("range_header", [
    "bytes=20-", "bytes=6-5", "bytes=-0", "bytes=-", "bytes=0-1,4-5", "items=0-1", "bytes=abc-def",
])
async def test_invalid_or_unsatisfiable_range_never_reads_blob(video_gallery, range_header):
    _, store, media = video_gallery
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=media_app(media))) as client:
        response = await client.get(media.link("video-1")["url"], headers={"Range": range_header})
    assert response.status_code == 416
    assert response.headers["content-range"] == "bytes */20"
    assert not any(a[0] == "bytes" for a in store.accesses)


@pytest.mark.anyio
async def test_unverifiable_if_range_returns_complete_asset(video_gallery):
    _, store, media = video_gallery
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=media_app(media))) as client:
        response = await client.get(media.link("video-1")["url"],
                                    headers={"Range": "bytes=0-1", "If-Range": '"unknown-etag"'})
    assert response.status_code == 200
    assert response.content == store.assets["playback"][0]


@pytest.mark.anyio
@pytest.mark.parametrize("method", ["GET", "HEAD"])
@pytest.mark.parametrize("change", ["gallery", "video", "variant", "signature", "expires", "missing", "duplicate"])
async def test_tampered_video_signature_fails_before_any_store_access(video_gallery, method, change):
    _, store, media = video_gallery
    parsed = urlsplit(media.link("video-1")["url"])
    path, query = parsed.path, parse_qs(parsed.query)
    if change == "gallery":
        path = path.replace("/local-videos/", "/other-gallery/")
    elif change == "video":
        path = path.replace("/video-1/", "/video-2/")
    elif change == "variant":
        path = path.replace("/playback", "/original")
    elif change == "signature":
        query["sig"] = ["0" * 64]
    elif change == "expires":
        query["expires"] = [str(int(query["expires"][0]) + 1)]
    elif change == "missing":
        query.pop("sig")
    else:
        query["sig"] *= 2
    url = urlunsplit(parsed._replace(path=path, query=urlencode(query, doseq=True)))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=media_app(media))) as client:
        response = await client.request(method, url)
    assert response.status_code == 403
    assert store.accesses == []


@pytest.mark.anyio
async def test_valid_video_signature_expires_before_store_access(video_gallery, monkeypatch):
    _, store, media = video_gallery
    now = 1_800_000_000
    monkeypatch.setattr("vooy_gallery.video_media.time.time", lambda: now)
    link = media.link("video-1")["url"]
    monkeypatch.setattr("vooy_gallery.video_media.time.time", lambda: now + media.ttl)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=media_app(media))) as client:
        response = await client.get(link)
    assert response.status_code == 403
    assert store.accesses == []


@pytest.mark.anyio
async def test_photo_signature_cannot_authorize_video_asset(video_gallery):
    settings, store, media = video_gallery
    # original is a common variant: differing HMAC domains must still reject this.
    photo_url = PhotoMedia(settings, store).link("video-1", "original")["url"]
    url = photo_url.replace("/media/", "/video-media/")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=media_app(media))) as client:
        response = await client.get(url)
    assert response.status_code == 403
    assert store.accesses == []


@pytest.mark.anyio
@pytest.mark.parametrize("segment_id,code", [("missing", "segment_not_found"), ("segment-2", "segment_not_in_video")])
async def test_selected_segment_must_belong_to_requested_video(video_gallery, segment_id, code):
    settings, store, media = video_gallery
    server = video_server(settings, store, media)
    async with create_connected_server_and_client_session(server) as client:
        result = await client.call_tool("get_video", {"video_id": "video-1", "segment_id": segment_id})
    assert result.isError
    assert result.structuredContent["code"] == code
    assert "video_url" not in result.structuredContent
    assert not any(a[0] in {"info", "bytes"} for a in store.accesses)


@pytest.mark.anyio
async def test_segment_bounds_are_checked_against_video_duration(video_gallery):
    settings, store, media = video_gallery
    store.segments["segment-1"]["end_ms"] = 10000
    server = video_server(settings, store, media)
    async with create_connected_server_and_client_session(server) as client:
        result = await client.call_tool("get_video", {"video_id": "video-1", "segment_id": "segment-1"})
    assert result.isError
    assert result.structuredContent["code"] == "segment_not_in_video"


@pytest.mark.anyio
@pytest.mark.parametrize("status", ["no_match", "needs_clarification"])
async def test_empty_or_unclear_search_has_no_media_results(video_gallery, status):
    settings, store, media = video_gallery
    server = video_server(settings, store, media, VideoSearchDouble(status=status))
    async with create_connected_server_and_client_session(server) as client:
        result = await client.call_tool("search_videos", {"query": "찾을 영상"})
    assert not result.isError
    assert result.structuredContent["status"] == status
    assert result.structuredContent["results"] == []
    assert all(c.type == "text" for c in result.content)
    assert store.accesses == []


@pytest.mark.anyio
async def test_video_tool_gallery_errors_and_missing_media_configuration(video_gallery):
    settings, store, _ = video_gallery
    server = video_server(replace(settings, mcp_public_base_url=None), store, None)
    async with create_connected_server_and_client_session(server) as client:
        for tool, arguments, code in [
            ("search_videos", {"query": ""}, "invalid_query"),
            ("search_videos", {"query": "영상", "limit": 4}, "invalid_limit"),
            ("search_videos", {"query": "영상", "gallery_id": "other"}, "gallery_not_available"),
            ("get_video", {"video_id": "video-1", "gallery_id": "other"}, "gallery_not_available"),
            ("get_video", {"video_id": "video-1"}, "media_url_unavailable"),
        ]:
            result = await client.call_tool(tool, arguments)
            assert result.isError
            assert result.structuredContent["code"] == code
        result = await client.call_tool("search_videos", {"query": "걷는 영상"})
        assert not result.isError
        assert "poster_url" not in result.structuredContent["results"][0]
        assert any(c.type == "image" for c in result.content)


@pytest.mark.anyio
async def test_search_result_segment_cannot_reference_another_video(video_gallery):
    settings, store, media = video_gallery
    server = video_server(settings, store, media, VideoSearchDouble(segment_id="segment-2"))
    async with create_connected_server_and_client_session(server) as client:
        result = await client.call_tool("search_videos", {"query": "걷는 영상"})
    assert result.isError
    assert result.structuredContent["code"] == "invalid_search_result"


@pytest.mark.anyio
async def test_video_failure_messages_never_expose_private_storage_details(video_gallery):
    settings, store, media = video_gallery

    class BrokenSearch:
        def search(self, *args, **kwargs):
            raise RuntimeError("credential-secret /private/video.mov")

    server = video_server(settings, store, media, BrokenSearch())
    async with create_connected_server_and_client_session(server) as client:
        result = await client.call_tool("search_videos", {"query": "영상"})
        assert result.isError
        assert result.structuredContent["code"] == "search_failed"
        assert "credential-secret" not in result.model_dump_json()
        assert "/private/" not in result.model_dump_json()
        missing = await client.call_tool("get_video", {"video_id": "missing"})
        assert missing.isError
        assert missing.structuredContent["code"] == "video_not_found"


@pytest.mark.parametrize("path", [
    "/video-media/local-videos/video-1/unknown", "/video-media/local-videos/video-1/playback/extra",
    "/video-media/local-videos/../playback", "/video-media/local-videos/a.mov/playback",
])
def test_video_signed_path_match_is_strict(path):
    assert not VideoMedia.matches_path(path)


@pytest.mark.anyio
async def test_full_server_video_auth_transport_and_media_routes(video_gallery):
    settings, video_store, _ = video_gallery
    settings = replace(settings, gallery_id="bts-daily", video_enabled=True, video_gallery_id="local-videos")
    search = VideoSearchDouble()

    class UnusedPhotoStore:
        def __getattr__(self, name):
            raise AssertionError(f"Video request accessed photo store: {name}")

    server = build_server(settings, store=UnusedPhotoStore(), video_store=video_store, video_search_service=search)
    app = server.streamable_http_app()
    tunnel_host = {"Host": "127.0.0.1:8765"}
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), headers={
            **tunnel_host, "Authorization": f"Bearer {settings.mcp_access_token}",
        }) as authenticated,
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), headers=tunnel_host) as receiver,
        streamable_http_client("http://localhost:8765/mcp", http_client=authenticated) as streams,
        ClientSession(streams[0], streams[1]) as client,
    ):
        await client.initialize()
        tools = await client.list_tools()
        assert {t.name for t in tools.tools} == {"search_photos", "get_photo", "search_videos", "get_video"}
        found = await client.call_tool("search_videos", {"query": "걷는 영상"})
        assert found.structuredContent["gallery_id"] == "local-videos"
        selected = await client.call_tool("get_video", {"video_id": "video-1", "segment_id": "segment-1"})
        assert not selected.isError
        url = selected.structuredContent["video_url"]
        denied_mcp = await receiver.post("http://localhost:8765/mcp", json={})
        assert denied_mcp.status_code == 401
        video_store.accesses.clear()
        unsigned = await receiver.get(urlunsplit(urlsplit(url)._replace(query="")))
        assert unsigned.status_code == 403
        assert video_store.accesses == []
        bad_host = await receiver.get(url, headers={"Host": "untrusted.example"})
        assert bad_host.status_code == 421
        assert video_store.accesses == []
        bad_origin = await receiver.get(url, headers={"Origin": "https://untrusted.example"})
        assert bad_origin.status_code == 403
        assert video_store.accesses == []
        playable = await receiver.get(url)
        assert playable.status_code == 200
        assert playable.headers["content-type"] == "video/mp4"
        assert playable.content == video_store.assets["playback"][0]
        partial = await receiver.get(url, headers={"Range": "bytes=0-3"})
        assert partial.status_code == 206
        assert partial.content == b"0123"
        assert selected.structuredContent["delivery_status"] == "not_sent"
    assert search.calls == [("걷는 영상", "local-videos", 3)]
