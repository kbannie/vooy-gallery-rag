import base64
import hashlib
from copy import deepcopy
from dataclasses import replace
from io import BytesIO
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.memory import create_connected_server_and_client_session
from PIL import Image

from vooy_gallery.mcp_server import build_server
from vooy_gallery.settings import Settings


class BlobStore:
    """MCP contract double: image bytes survive deletion or mutation of local cache files."""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True)
        self.photos = {}
        self.assets = {}

    def upsert_photo(self, photo):
        self.photos[photo["photo_id"]] = deepcopy(photo)
        for thumbnail, key in [(False, "image_path"), (True, "thumbnail_path")]:
            self.assets[photo["photo_id"], thumbnail] = (
                (self.data_dir / photo[key]).read_bytes(), photo["mime_type"]
            )

    def get_photo(self, photo_id, gallery_id="bts-demo"):
        photo = self.photos.get(photo_id)
        return deepcopy(photo) if photo and photo["gallery_id"] == gallery_id else None

    def get_asset(self, photo_id, gallery_id="bts-demo", thumbnail=False):
        if not self.get_photo(photo_id, gallery_id):
            raise ValueError("photo not found")
        return self.assets[photo_id, thumbnail]


class FakeSearch:
    def __init__(self, photo_id="photo-1", status="ok"):
        self.photo_id = photo_id
        self.status = status
        self.calls = []

    def search(self, query, gallery_id, limit):
        self.calls.append((query, gallery_id, limit))
        return {
            "result_id": "result-1", "query": query, "gallery_id": gallery_id,
            "status": self.status, "applied_filters": {"location": "서울"},
            "clarification": "어느 날짜인가요?" if self.status == "needs_clarification" else None,
            "results": [{
                "photo_id": self.photo_id, "caption": "무대에서 노래하는 사람들", "score": 0.8,
                "reason": "공연 장면", "thumbnail_path": "secret/local/path.jpg",
            }] if self.status == "ok" else [],
            "internal_path": "/should/not/be/exposed",
        }


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def gallery(tmp_path):
    data_dir = tmp_path / "data"
    store = BlobStore(data_dir)
    (data_dir / "images").mkdir()
    (data_dir / "thumbnails").mkdir()
    image_path = data_dir / "images/photo.png"
    Image.new("RGB", (32, 24), "red").save(image_path)
    image_bytes = image_path.read_bytes()
    (data_dir / "thumbnails/photo.png").write_bytes(image_bytes)
    photo = {
        "photo_id": "photo-1", "gallery_id": "bts-demo",
        "image_path": "images/photo.png", "thumbnail_path": "thumbnails/photo.png",
        "mime_type": "image/png", "metadata": {
            "origin": "synthetic", "location": "서울", "captured_at": "2026-10-10T12:00:00+09:00",
        },
        "source": {
            "attribution": "Example author / CC BY 4.0",
            "page_url": "https://commons.wikimedia.org/wiki/File:Example.png",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
        },
    }
    store.upsert_photo(photo)
    return Settings(data_dir=data_dir), store, photo, image_bytes


@pytest.mark.anyio
async def test_protocol_search_and_selected_original(gallery):
    settings, store, _, original_bytes = gallery
    search = FakeSearch()
    server = build_server(settings, store, search)
    async with create_connected_server_and_client_session(server, raise_exceptions=True) as client:
        tools = await client.list_tools()
        assert {tool.name for tool in tools.tools} == {"search_photos", "get_photo"}
        result = await client.call_tool("search_photos", {"query": "서울 공연 사진"})
        assert not result.isError
        public = result.structuredContent
        assert public["status"] == "ok"
        assert public["result_id"] == "result-1"
        assert "thumbnail_path" not in public["results"][0]
        assert "internal_path" not in public
        assert "secret/local/path" not in result.model_dump_json()
        preview = result.content[public["results"][0]["preview_content_index"]]
        with Image.open(BytesIO(base64.b64decode(preview.data))) as image:
            assert image.size == (32, 24)
        selected = await client.call_tool("get_photo", {"photo_id": public["results"][0]["photo_id"]})
        assert not selected.isError
        assert selected.structuredContent["delivery_status"] == "not_sent"
        assert selected.structuredContent["metadata"]["origin"] == "synthetic"
        assert "attribution" not in selected.structuredContent
        assert base64.b64decode(selected.content[1].data) == original_bytes
        assert str(store.data_dir) not in selected.model_dump_json()
    assert search.calls == [("서울 공연 사진", "bts-demo", 3)]


@pytest.mark.anyio
async def test_omitted_gallery_uses_configured_daily_gallery(gallery):
    settings, store, photo, original_bytes = gallery
    settings = replace(settings, gallery_id="bts-daily")
    photo["gallery_id"] = "bts-daily"
    store.upsert_photo(photo)
    search = FakeSearch()
    server = build_server(settings, store, search)
    async with create_connected_server_and_client_session(server) as client:
        result = await client.call_tool("search_photos", {"query": "산책 사진"})
        assert not result.isError
        assert result.structuredContent["gallery_id"] == "bts-daily"
        selected = await client.call_tool("get_photo", {"photo_id": "photo-1"})
        assert not selected.isError
        assert base64.b64decode(selected.content[1].data) == original_bytes
    assert search.calls == [("산책 사진", "bts-daily", 3)]


@pytest.mark.anyio
@pytest.mark.parametrize("status", ["no_match", "needs_clarification"])
async def test_non_matches_have_no_image_blocks(gallery, status):
    settings, store, _, _ = gallery
    server = build_server(settings, store, FakeSearch(status=status))
    async with create_connected_server_and_client_session(server) as client:
        result = await client.call_tool("search_photos", {"query": "조건에 없는 사진"})
        assert not result.isError
        assert result.structuredContent["status"] == status
        assert result.structuredContent["results"] == []
        assert all(content.type == "text" for content in result.content)


@pytest.mark.anyio
async def test_gallery_scope_missing_photo_and_invalid_query(gallery):
    settings, store, _, _ = gallery
    search = FakeSearch()
    server = build_server(settings, store, search)
    async with create_connected_server_and_client_session(server) as client:
        for tool, arguments, code in [
            ("get_photo", {"photo_id": "photo-1", "gallery_id": "private"}, "gallery_not_available"),
            ("get_photo", {"photo_id": "missing"}, "photo_not_found"),
            ("search_photos", {"query": "BTS", "gallery_id": "private"}, "gallery_not_available"),
            ("search_photos", {"query": " "}, "invalid_query"),
            ("search_photos", {"query": "BTS", "limit": 4}, "invalid_limit"),
        ]:
            result = await client.call_tool(tool, arguments)
            assert result.isError
            assert result.structuredContent["code"] == code
    assert search.calls == []


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["corrupt", "oversized"])
async def test_invalid_database_assets_never_leave_server(gallery, monkeypatch, mode):
    settings, store, _, _ = gallery
    if mode == "corrupt":
        store.assets["photo-1", False] = (b"not an image", "image/png")
    else:
        monkeypatch.setattr("vooy_gallery.mcp_server.MAX_IMAGE_BYTES", 1)
    server = build_server(settings, store, FakeSearch())
    async with create_connected_server_and_client_session(server) as client:
        result = await client.call_tool("get_photo", {"photo_id": "photo-1"})
        assert result.isError
        assert result.structuredContent["code"] == "image_unavailable"
        assert all(content.type != "image" for content in result.content)


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["deleted", "corrupt", "outside_pointer"])
async def test_get_photo_uses_database_bytes_when_local_cache_is_unavailable(gallery, tmp_path, mode):
    settings, store, _, original = gallery
    path = store.data_dir / "images/photo.png"
    if mode == "deleted":
        path.unlink()
        (store.data_dir / "thumbnails/photo.png").unlink()
    elif mode == "corrupt":
        path.write_bytes(b"untrusted local cache")
    else:
        outside = tmp_path / "private.txt"
        outside.write_text("private file should never be read")
        store.photos["photo-1"]["image_path"] = str(outside)
    server = build_server(settings, store, FakeSearch())
    async with create_connected_server_and_client_session(server) as client:
        result = await client.call_tool("get_photo", {"photo_id": "photo-1"})
        assert not result.isError
        assert base64.b64decode(result.content[1].data) == original
        assert "private file" not in result.model_dump_json()
        preview = await client.call_tool("search_photos", {"query": "서울 사진"})
        assert not preview.structuredContent["results"][0].get("preview_unavailable")


def test_local_server_rejects_public_binding(gallery):
    settings, store, _, _ = gallery
    with pytest.raises(ValueError, match="loopback"):
        build_server(Settings(data_dir=settings.data_dir, mcp_host="0.0.0.0"), store, FakeSearch())


@pytest.mark.anyio
async def test_provider_failure_hides_sensitive_details(gallery):
    settings, store, _, _ = gallery

    class BrokenSearch:
        def search(self, *args, **kwargs):
            raise RuntimeError("/private/key-file credential-secret")

    server = build_server(settings, store, BrokenSearch())
    async with create_connected_server_and_client_session(server) as client:
        result = await client.call_tool("search_photos", {"query": "BTS"})
        assert result.isError
        assert result.structuredContent["code"] == "search_failed"
        assert "credential-secret" not in result.model_dump_json()


@pytest.mark.anyio
@pytest.mark.parametrize("access_token", [None, "synthetic-mcp-test-token"])
async def test_streamable_http_round_trip(gallery, access_token, caplog):
    """Exercise the public /mcp transport using the official SDK client."""
    settings, store, _, original_bytes = gallery
    settings = replace(settings, mcp_access_token=access_token)
    server = build_server(settings, store, FakeSearch())
    app = server.streamable_http_app()
    headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), headers=headers) as http_client,
        streamable_http_client("http://localhost:8765/mcp", http_client=http_client) as streams,
        ClientSession(streams[0], streams[1]) as client,
    ):
        initialized = await client.initialize()
        assert initialized.serverInfo.name == "BTS Gallery"
        tools = await client.list_tools()
        assert {tool.name for tool in tools.tools} == {"search_photos", "get_photo"}
        candidates = await client.call_tool("search_photos", {"query": "서울 공연"})
        assert candidates.structuredContent["status"] == "ok"
        photo_id = candidates.structuredContent["results"][0]["photo_id"]
        selected = await client.call_tool("get_photo", {"photo_id": photo_id})
        assert base64.b64decode(selected.content[1].data) == original_bytes
        if access_token:
            assert access_token not in selected.model_dump_json()
    if access_token:
        assert access_token not in caplog.text


@pytest.mark.anyio
@pytest.mark.parametrize("authorization", [None, "Bearer wrong-token", "Basic wrong-token"])
async def test_http_requires_configured_bearer_before_running_tools(gallery, authorization, caplog):
    settings, store, _, _ = gallery
    token = "synthetic-mcp-test-token"
    settings = replace(settings, mcp_access_token=token)
    search = FakeSearch()
    server = build_server(settings, store, search)
    app = server.streamable_http_app()
    headers = {"Authorization": authorization} if authorization else {}
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client,
    ):
        response = await client.post("http://localhost:8765/mcp", headers=headers, json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "search_photos", "arguments": {"query": "서울 사진"}},
        })
        assert response.status_code == 401
        assert response.headers["www-authenticate"].startswith("Bearer ")
        assert token not in response.text
    assert search.calls == []
    assert token not in caplog.text


@pytest.mark.anyio
@pytest.mark.parametrize("headers, expected_status", [
    ({"Host": "untrusted.example"}, 421),
    ({"Origin": "https://untrusted.example"}, 403),
])
async def test_bearer_keeps_loopback_host_and_origin_protection(gallery, headers, expected_status):
    settings, store, _, _ = gallery
    token = "synthetic-mcp-test-token"
    server = build_server(replace(settings, mcp_access_token=token), store, FakeSearch())
    app = server.streamable_http_app()
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client,
    ):
        response = await client.post("http://localhost:8765/mcp", headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json, text/event-stream",
            **headers,
        }, json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
        assert response.status_code == expected_status


@pytest.fixture
def media_gallery(gallery):
    settings, store, photo, original = gallery
    settings = replace(
        settings,
        mcp_access_token="synthetic-media-access-token",
        mcp_public_base_url="https://gallery.example.com",
        mcp_image_url_ttl_seconds=900,
    )
    return settings, store, photo, original


@pytest.mark.anyio
async def test_mcp_signed_links_fetch_original_and_thumbnail_without_bearer(media_gallery, caplog):
    settings, store, _, original = media_gallery
    server = build_server(settings, store, FakeSearch())
    app = server.streamable_http_app()
    tunnel_host = {"Host": "127.0.0.1:8765"}
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), headers={
            **tunnel_host, "Authorization": f"Bearer {settings.mcp_access_token}",
        }) as mcp_http,
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), headers=tunnel_host) as image_http,
        streamable_http_client("http://localhost:8765/mcp", http_client=mcp_http) as streams,
        ClientSession(streams[0], streams[1]) as client,
    ):
        await client.initialize()
        result = await client.call_tool("get_photo", {"photo_id": "photo-1"})
        assert not result.isError
        selected = result.structuredContent
        assert selected["delivery_status"] == "not_sent"
        assert selected["byte_size"] == len(original)
        assert selected["image_sha256"] == hashlib.sha256(original).hexdigest()
        assert selected["filename"].endswith(".png")
        assert selected["image_url_expires_at"]
        assert base64.b64decode(result.content[1].data) == original
        assert result.content[2].type == "resource_link"
        assert str(result.content[2].uri) == selected["image_url"]
        assert settings.mcp_access_token not in result.model_dump_json()
        assert str(store.data_dir) not in result.model_dump_json()

        original_response = await image_http.get(selected["image_url"])
        assert original_response.status_code == 200
        assert original_response.content == original
        assert original_response.headers["content-type"] == "image/png"
        assert "no-store" in original_response.headers["cache-control"]
        head = await image_http.head(selected["image_url"])
        assert head.status_code == 200
        assert head.content == b""
        assert int(head.headers["content-length"]) == len(original)

        candidates = await client.call_tool("search_photos", {"query": "서울 사진"})
        candidate = candidates.structuredContent["results"][0]
        assert candidate["preview_url_expires_at"]
        preview = await image_http.get(candidate["preview_url"])
        assert preview.status_code == 200
        assert preview.content == store.assets["photo-1", True][0]
        assert settings.mcp_access_token not in candidates.model_dump_json()

        denied = await image_http.post("http://localhost:8765/mcp", json={})
        assert denied.status_code == 401
        other_route = await image_http.get("https://gallery.example.com/private")
        assert other_route.status_code == 401
    assert settings.mcp_access_token not in caplog.text


@pytest.mark.anyio
@pytest.mark.parametrize("method", ["GET", "HEAD"])
@pytest.mark.parametrize("change", ["gallery", "photo", "variant", "signature", "missing_signature"])
async def test_tampered_media_links_fail_before_store_access(media_gallery, monkeypatch, method, change):
    from vooy_gallery.media import PhotoMedia

    settings, store, _, _ = media_gallery
    link = PhotoMedia(settings, store).link("photo-1")["url"]
    parsed = urlsplit(link)
    path, query = parsed.path, parse_qs(parsed.query)
    if change == "gallery":
        path = path.replace("/bts-demo/", "/other-gallery/")
    elif change == "photo":
        path = path.replace("/photo-1/", "/photo-2/")
    elif change == "variant":
        path = path.replace("/original", "/thumbnail")
    elif change == "signature":
        query["sig"] = ["0" * 64]
    else:
        query.pop("sig")
    tampered = urlunsplit(parsed._replace(path=path, query=urlencode(query, doseq=True)))
    accesses = []

    def forbidden(*args, **kwargs):
        accesses.append(True)
        raise AssertionError("Invalid signature reached the database")

    monkeypatch.setattr(store, "get_photo", forbidden)
    monkeypatch.setattr(store, "get_asset", forbidden)
    server = build_server(settings, store, FakeSearch())
    app = server.streamable_http_app()
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), headers={"Host": "127.0.0.1:8765"}) as client,
    ):
        response = await client.request(method, tampered)
        assert response.status_code == 403
    assert accesses == []


@pytest.mark.anyio
@pytest.mark.parametrize("path", [
    "/media/bts-demo/photo-1/unlisted",
    "/media/bts-demo/photo-1/original/extra",
])
async def test_media_bearer_exception_matches_only_exact_routes(media_gallery, path):
    settings, store, _, _ = media_gallery
    server = build_server(settings, store, FakeSearch())
    app = server.streamable_http_app()
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client,
    ):
        response = await client.get(f"http://localhost:8765{path}")
        assert response.status_code == 401


@pytest.mark.anyio
@pytest.mark.parametrize("method", ["GET", "HEAD"])
async def test_expired_media_link_fails_before_store_access(media_gallery, monkeypatch, method):
    from vooy_gallery.media import PhotoMedia

    settings, store, _, _ = media_gallery
    issued_at = 1_800_000_000
    monkeypatch.setattr("vooy_gallery.media.time.time", lambda: issued_at)
    link = PhotoMedia(settings, store).link("photo-1")["url"]
    monkeypatch.setattr("vooy_gallery.media.time.time", lambda: issued_at + settings.mcp_image_url_ttl_seconds)
    accesses = []

    def forbidden(*args, **kwargs):
        accesses.append(True)
        raise AssertionError("Expired signature reached the database")

    monkeypatch.setattr(store, "get_photo", forbidden)
    monkeypatch.setattr(store, "get_asset", forbidden)
    server = build_server(settings, store, FakeSearch())
    app = server.streamable_http_app()
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), headers={"Host": "127.0.0.1:8765"}) as client,
    ):
        response = await client.request(method, link)
        assert response.status_code == 403
    assert accesses == []


@pytest.mark.anyio
@pytest.mark.parametrize("extra_headers, status", [
    ({"Host": "untrusted.example"}, 421),
    ({"Origin": "https://untrusted.example"}, 403),
])
async def test_signed_media_keeps_host_and_origin_protection(media_gallery, extra_headers, status):
    from vooy_gallery.media import PhotoMedia

    settings, store, _, _ = media_gallery
    link = PhotoMedia(settings, store).link("photo-1")["url"]
    server = build_server(settings, store, FakeSearch())
    app = server.streamable_http_app()
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client,
    ):
        response = await client.get(link, headers={"Host": "127.0.0.1:8765", **extra_headers})
        assert response.status_code == status


@pytest.mark.anyio
async def test_signed_media_link_does_not_authorize_post_requests(media_gallery):
    from vooy_gallery.media import PhotoMedia

    settings, store, _, _ = media_gallery
    link = PhotoMedia(settings, store).link("photo-1")["url"]
    server = build_server(settings, store, FakeSearch())
    app = server.streamable_http_app()
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), headers={"Host": "127.0.0.1:8765"}) as client,
    ):
        response = await client.post(link)
        assert response.status_code == 401


def test_public_media_url_requires_token(gallery):
    settings, store, _, _ = gallery
    with pytest.raises(ValueError):
        build_server(replace(settings, mcp_public_base_url="https://gallery.example.com"), store, FakeSearch())


@pytest.mark.parametrize("base", [
    "http://gallery.example.com", "https://gallery.example.com/path",
    "https://user:password@gallery.example.com", "https://gallery.example.com?token=value",
])
def test_media_requires_https_origin_without_embedded_credentials(media_gallery, base):
    settings, store, _, _ = media_gallery
    with pytest.raises(ValueError):
        build_server(replace(settings, mcp_public_base_url=base), store, FakeSearch())


@pytest.mark.parametrize("ttl", [59, 3601])
def test_media_expiry_limits_are_enforced(media_gallery, ttl):
    settings, store, _, _ = media_gallery
    with pytest.raises(ValueError):
        build_server(replace(settings, mcp_image_url_ttl_seconds=ttl), store, FakeSearch())
