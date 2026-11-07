import hashlib
from io import BytesIO

import httpx
from PIL import Image

from vooy_gallery.collector import (
    CommonsClient,
    accepted_license,
    collect,
    iter_candidates,
    source_identities,
)


class MemoryStore:
    def __init__(self, path):
        self.data_dir = path
        self.photos = {}

    def list_photos(self, gallery_id="bts-demo"):
        return [photo for photo in self.photos.values() if photo["gallery_id"] == gallery_id]

    def upsert_photo(self, photo):
        self.photos[photo["photo_id"]] = photo


def test_licenses_and_explicit_identity_only():
    assert accepted_license("CC BY-SA 4.0", "")
    assert accepted_license("CC0", "")
    assert accepted_license("Public domain", "")
    assert not accepted_license("CC BY-NC 4.0", "https://creativecommons.org/licenses/by-nc/4.0/")
    assert not accepted_license("All rights reserved", "")
    assert source_identities("File:Rome V.jpg", "a description", []) == []
    assert source_identities("File:BTS Live Vol. V.jpg", "arriving on Jin Air", []) == []
    assert source_identities("File:BTS V at an event.jpg", "", []) == ["V"]
    assert source_identities("File:BTS Jimin.jpg", "", []) == ["Jimin"]
    assert source_identities("File:stage.jpg", "", ["Category:V (vocalist) in 2019"]) == ["V"]


def test_category_continuation_and_repeated_files():
    seen_continuations = []

    def handler(request):
        params = dict(request.url.params)
        seen_continuations.append(params.get("cmcontinue"))
        if not params.get("cmcontinue"):
            return httpx.Response(200, json={"query": {"categorymembers": [{"pageid": 11, "ns": 6, "title": "File:BTS.jpg"}]},
                                             "continue": {"cmcontinue": "next", "continue": "-||"}})
        return httpx.Response(200, json={"query": {"categorymembers": [{"pageid": 11, "ns": 6, "title": "File:BTS.jpg"},
                                                                                       {"pageid": 12, "ns": 6, "title": "File:BTS logo.svg"}]}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        api = CommonsClient(client, delay=0)
        candidates = list(iter_candidates(api, ("Category:BTS by year",), 10, 10))
    assert [item["pageid"] for item in candidates] == [11]
    assert seen_continuations == [None, "next"]


def test_retry_maxlag_and_429(monkeypatch):
    monkeypatch.setattr("vooy_gallery.collector.time.sleep", lambda _: None)
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        if len(calls) == 2:
            return httpx.Response(200, json={"error": {"code": "maxlag", "info": "Lagged"}})
        return httpx.Response(200, json={"query": {"ok": True}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        api = CommonsClient(client, delay=0)
        assert api.api(meta="siteinfo")["query"]["ok"]
    assert len(calls) == 3
    assert "VooyGalleryDemo" in calls[0].headers["User-Agent"]


def test_standard_library_transport_keeps_application_identity(monkeypatch):
    class Upstream:
        status = 200

        def __init__(self):
            self.headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b'{"query":{"ok":true}}'

    captured = []

    def open_request(request, timeout):
        captured.append(request)
        assert timeout == 30
        return Upstream()

    monkeypatch.setattr("vooy_gallery.collector.urlopen", open_request)
    api = CommonsClient(delay=0)
    try:
        assert api.api(meta="siteinfo")["query"]["ok"]
    finally:
        api.close()
    assert "VooyGalleryDemo" in captured[0].get_header("User-agent")
    assert "maxlag=5" in captured[0].full_url


def test_retry_after_is_not_shortened_and_is_reported(monkeypatch):
    pauses = []
    monkeypatch.setattr("vooy_gallery.collector.time.sleep", pauses.append)
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        if count == 1:
            return httpx.Response(429, headers={"Retry-After": "46"})
        return httpx.Response(200, json={"query": {"ok": True}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        api = CommonsClient(client, delay=0)
        assert api.api(meta="siteinfo")["query"]["ok"]
    assert pauses == [46.0]
    assert api.telemetry["requests"] == 2
    assert api.telemetry["rate_limits"] == [{"host": "commons.wikimedia.org", "retry_after": "46", "pause_seconds": 46.0}]


def test_collection_preserves_original_deduplicates_and_resumes(tmp_path):
    buffer = BytesIO()
    Image.new("RGB", (640, 480), (80, 120, 160)).save(buffer, "JPEG")
    original = buffer.getvalue()
    requests = []

    def handler(request):
        requests.append(request)
        params = dict(request.url.params)
        if request.url.host == "upload.wikimedia.org":
            return httpx.Response(200, content=original)
        if params.get("list") == "categorymembers":
            return httpx.Response(200, json={"query": {"categorymembers": [
                {"pageid": i, "ns": 6, "title": f"File:BTS photo {i}.jpg"} for i in (1, 2)]}})
        pages = []
        for value in params["pageids"].split("|"):
            ident = int(value)
            pages.append({"pageid": ident, "title": f"File:BTS photo {ident}.jpg", "imageinfo": [{
                "mime": "image/jpeg", "width": 640, "height": 480, "size": len(original),
                "url": f"https://upload.wikimedia.org/photograph-{ident}.jpg",
                "descriptionurl": f"https://commons.wikimedia.org/wiki/File:BTS_photo_{ident}.jpg",
                "extmetadata": {"LicenseShortName": {"value": "CC BY 4.0"},
                                "LicenseUrl": {"value": "https://creativecommons.org/licenses/by/4.0/"},
                                "Artist": {"value": "<a>Photographer</a>"},
                                "ImageDescription": {"value": "BTS on stage"}}}]})
        return httpx.Response(200, json={"query": {"pages": pages}})

    store = MemoryStore(tmp_path)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        stats = collect(store, target=2, roots=("Category:BTS by year",), client=client, delay=0)
        assert stats["status"] == "incomplete"
        assert stats["downloaded"] == 1
        assert stats["skipped"]["exact_duplicate"] == 1
        photo = next(iter(store.photos.values()))
        assert (tmp_path / photo["image_path"]).read_bytes() == original
        assert photo["sha256"] == hashlib.sha256(original).hexdigest()
        assert photo["source"]["author"] == "Photographer"
        assert photo["metadata"]["origin"] == "synthetic"
        before = len(requests)
        resumed = collect(store, target=1, roots=("Category:BTS by year",), client=client, delay=0)
        assert resumed["downloaded"] == 0
        assert len(requests) == before


def test_corrupt_download_is_not_saved(tmp_path):
    def handler(request):
        params = dict(request.url.params)
        if request.url.host == "upload.wikimedia.org":
            return httpx.Response(200, content=b"not an image")
        if params.get("list"):
            return httpx.Response(200, json={"query": {"categorymembers": [{"pageid": 1, "ns": 6, "title": "File:BTS.jpg"}]}})
        return httpx.Response(200, json={"query": {"pages": [{"pageid": 1, "title": "File:BTS.jpg", "imageinfo": [{
            "mime": "image/jpeg", "width": 600, "height": 600, "url": "https://upload.wikimedia.org/bts.jpg",
            "extmetadata": {"LicenseShortName": {"value": "CC0"}}}]}]}})

    store = MemoryStore(tmp_path)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        stats = collect(store, target=1, client=client, delay=0, roots=("Category:BTS by year",))
    assert stats["total"] == 0
    assert stats["skipped"]["download_or_decode_error"] == 1
    assert not store.photos


def test_top_up_does_not_count_or_download_excluded_photos(tmp_path):
    buffer = BytesIO()
    Image.new("RGB", (640, 480), (200, 90, 40)).save(buffer, "JPEG")
    payload = buffer.getvalue()
    store = MemoryStore(tmp_path)
    # A database record can outlive an evicted local cache. Exclusion still holds.
    store.upsert_photo({"photo_id": "commons-1", "gallery_id": "bts-demo", "image_path": "missing.jpg",
                        "sha256": "excluded-original", "index_status": "excluded"})
    downloaded_urls = []

    def handler(request):
        params = dict(request.url.params)
        if request.url.host == "upload.wikimedia.org":
            downloaded_urls.append(str(request.url))
            return httpx.Response(200, content=payload)
        if params.get("list"):
            return httpx.Response(200, json={"query": {"categorymembers": [
                {"pageid": 1, "ns": 6, "title": "File:BTS excluded.jpg"},
                {"pageid": 2, "ns": 6, "title": "File:BTS replacement.jpg"}]}})
        assert params["pageids"] == "2"
        return httpx.Response(200, json={"query": {"pages": [{"pageid": 2, "title": "File:BTS replacement.jpg", "imageinfo": [{
            "mime": "image/jpeg", "width": 640, "height": 480, "url": "https://upload.wikimedia.org/replacement.jpg",
            "extmetadata": {"LicenseShortName": {"value": "CC0"}}}]}]}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        stats = collect(store, target=1, client=client, delay=0, roots=("Category:BTS by year",))
    assert stats["status"] == "target_reached"
    assert stats["excluded"] == 1
    assert stats["existing"] == 0
    assert stats["total"] == 1
    assert stats["downloaded"] == 1
    assert stats["skipped"]["already_excluded"] == 1
    assert store.photos["commons-1"]["index_status"] == "excluded"
    assert downloaded_urls == ["https://upload.wikimedia.org/replacement.jpg"]


def test_database_photo_counts_without_a_local_cache(tmp_path):
    store = MemoryStore(tmp_path)
    store.upsert_photo({"photo_id": "commons-1", "gallery_id": "bts-demo", "image_path": "evicted.jpg",
                        "sha256": "database-original", "index_status": "indexed"})

    def handler(request):
        raise AssertionError("Existing PostgreSQL assets must not be downloaded again")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        stats = collect(store, target=1, client=client, delay=0)
    assert stats["status"] == "target_reached"
    assert stats["existing"] == 1
    assert stats["downloaded"] == 0
