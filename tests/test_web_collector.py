import hashlib
import json
from io import BytesIO
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request

import numpy as np
import pytest
from PIL import Image

from vooy_gallery.web_collector import (
    FetchFailure,
    PublicRedirectHandler,
    WebFetcher,
    collect_web,
    discover_image_urls,
    image_id,
    public_url,
)


class MemoryStore:
    def __init__(self, path):
        self.data_dir = path
        self.photos = {}

    def list_photos(self, gallery_id="bts-daily"):
        return [photo for photo in self.photos.values() if photo["gallery_id"] == gallery_id]

    def upsert_photo(self, photo):
        self.photos[photo["photo_id"]] = photo


class FakeFetcher:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.rate_limits = []

    def fetch(self, url, limit=20 * 1024 * 1024):
        self.calls.append(url)
        response = self.responses[url]
        if isinstance(response, Exception):
            raise response
        return response, {}, url


def jpeg(seed):
    array = np.random.default_rng(seed).integers(0, 255, (400, 500, 3), dtype=np.uint8)
    output = BytesIO()
    Image.fromarray(array).save(output, "JPEG")
    return output.getvalue()


def test_original_wordpress_srcset_and_koreaboo_lazyload():
    html = """<body><header><img src='/header.jpg'></header><article><div class='entry-content'>
    <figure><a href='/full.jpg'><img src='/small.jpg' alt='Jimin walking'></a><figcaption>A park walk</figcaption></figure>
    <img data-orig-file='/original.jpg' srcset='/medium.jpg 500w, /large.jpg 1500w' src='/tiny.jpg'>
    <img srcset='/small2.jpg 320w, /big2.jpg 1600w' src='/tiny2.jpg'>
    <img src='data:image/gif;base64,AAAA' data-src='//cdn.example.org/casual.webp' alt='casual clothes'>
    <img src='/original.jpg'><img src='/logo.png' alt='site logo'><img src='/animation.gif'>
    <img src='https://encrypted-tbn0.gstatic.com/images?q=thumb'>
    <aside><img src='/sidebar.jpg'></aside></div></article></body>"""
    found = discover_image_urls(html, "https://news.example.org/article")
    assert [item["url"] for item in found] == ["https://news.example.org/full.jpg", "https://news.example.org/original.jpg",
                                                "https://news.example.org/big2.jpg", "https://cdn.example.org/casual.webp"]
    assert found[0]["caption"] == "A park walk"
    assert found[0]["alt"] == "Jimin walking"


def test_signed_queries_are_preserved_and_ids_ignore_expiry():
    url = "https://cdn.example.org/image.jpg?w=900&Signature=a%2Bb%2Fc%3D&Expires=123&KeyName=foo&utm_source=x"
    normalized = public_url(url, url)
    assert normalized == "https://cdn.example.org/image.jpg?w=900&Signature=a%2Bb%2Fc%3D&Expires=123&KeyName=foo"
    assert image_id(normalized) == image_id(normalized.replace("Expires=123", "Expires=999").replace("a%2Bb%2Fc%3D", "new%2Bsig"))
    assert public_url("http://127.0.0.1/private.jpg", url) is None
    assert public_url("file:///etc/passwd", url) is None


def test_redirect_is_rejected_before_private_request():
    handler = PublicRedirectHandler({})
    with pytest.raises(FetchFailure, match="non_public_redirect"):
        handler.redirect_request(Request("https://news.example.org/a"), None, 302, "Found", {}, "http://127.0.0.1/x")


def test_host_cooldown_does_not_sleep_or_block_another_host(monkeypatch):
    sleeps = []
    monkeypatch.setattr("vooy_gallery.web_collector.time.sleep", sleeps.append)
    fetcher = WebFetcher()
    calls = []

    class Response:
        def __init__(self):
            self.headers = {"Content-Length": "2"}
            self.data = BytesIO(b"ok")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def geturl(self):
            return "https://other.example.org/image.jpg"

        def read(self, amount):
            return self.data.read(amount)

    def open_request(request, timeout):
        calls.append(request.full_url)
        if "limited.example.org" in request.full_url:
            raise HTTPError(request.full_url, 429, "Too Many Requests", {"Retry-After": "600"}, BytesIO())
        return Response()

    fetcher.opener = SimpleNamespace(open=open_request)
    with pytest.raises(FetchFailure, match="http_429"):
        fetcher.fetch("https://limited.example.org/image.jpg")
    with pytest.raises(FetchFailure, match="host_cooldown"):
        fetcher.fetch("https://limited.example.org/another.jpg")
    assert fetcher.fetch("https://other.example.org/image.jpg")[0] == b"ok"
    assert len(calls) == 2
    assert all(pause <= 1 for pause in sleeps)
    assert fetcher.rate_limits[0]["retry_after_seconds"] == 600


def test_collect_dedup_provenance_and_per_image_identity(tmp_path):
    article = "https://news.example.org/daily"
    other = "https://other.example.org/travel"
    first, second = jpeg(1), jpeg(2)
    sources = [{"page_url": article, "title": "Daily photos", "identities": ["Jimin", "Jung Kook"]},
               {"page_url": other, "title": "Travel photos", "identities": ["V"], "author": "Source author"}]
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(sources))
    fetcher = FakeFetcher({article: b"<article><img src='https://cdn.example.org/one.jpg' alt='Jimin in a park'><img src='https://cdn.example.org/duplicate.jpg'></article>",
                           other: b"<main><img src='https://cdn.example.org/two.jpg'></main>",
                           "https://cdn.example.org/one.jpg": first, "https://cdn.example.org/duplicate.jpg": first,
                           "https://cdn.example.org/two.jpg": second})
    store = MemoryStore(tmp_path)
    report = collect_web(store, path, target=2, fetcher=fetcher)
    assert report["status"] == "target_reached"
    assert report["skipped"]["exact_duplicate"] == 1
    photos = list(store.photos.values())
    assert photos[0]["identities"] == ["Jimin"]
    assert photos[0]["identity_source"] == "image_alt_or_caption"
    assert photos[1]["identities"] == ["V"]
    assert photos[0]["source"]["license_name"] == "Not specified"
    assert photos[0]["source"]["page_url"] == article
    assert photos[0]["metadata"]["origin"] == "synthetic"
    assert (tmp_path / photos[0]["image_path"]).read_bytes() == first
    assert photos[0]["sha256"] == hashlib.sha256(first).hexdigest()
    before = len(fetcher.calls)
    assert collect_web(store, path, target=2, fetcher=fetcher)["downloaded"] == 0
    assert len(fetcher.calls) == before


def test_multi_member_article_does_not_label_every_image(tmp_path):
    page = "https://news.example.org/group"
    path = tmp_path / "sources.json"
    path.write_text(json.dumps([{"page_url": page, "identities": ["Jin", "Jimin", "V"]}]))
    fetcher = FakeFetcher({page: b"<article><img src='/travel.jpg' alt='Travel photo'></article>",
                           "https://news.example.org/travel.jpg": jpeg(3)})
    store = MemoryStore(tmp_path)
    assert collect_web(store, path, target=1, fetcher=fetcher)["total"] == 1
    photo = next(iter(store.photos.values()))
    assert photo["identities"] == []
    assert photo["source"]["article_identities"] == ["Jin", "Jimin", "V"]


def test_rejected_records_do_not_count_and_are_not_downloaded_again(tmp_path):
    page = "https://news.example.org/group"
    first_url, next_url = "https://cdn.example.org/rejected.jpg", "https://cdn.example.org/new.jpg"
    path = tmp_path / "sources.json"
    path.write_text(json.dumps([{"page_url": page}]))
    store = MemoryStore(tmp_path)
    store.upsert_photo({"photo_id": image_id(first_url), "gallery_id": "bts-daily", "collection_theme": "daily",
                        "review": {"included": False, "reason": "stage photograph"}, "source": {"page_url": page}})
    fetcher = FakeFetcher({page: f"<article><img src='{first_url}'><img src='{next_url}'></article>".encode(), next_url: jpeg(4)})
    report = collect_web(store, path, target=1, fetcher=fetcher)
    assert report["existing"] == 0
    assert report["downloaded"] == 1
    assert first_url not in fetcher.calls


def test_per_source_cap_and_403_do_not_trigger_bypass(tmp_path):
    first, blocked, third = "https://news.example.org/a", "https://blocked.example.org/b", "https://other.example.org/c"
    path = tmp_path / "sources.json"
    path.write_text(json.dumps([{"page_url": url} for url in (first, blocked, third)]))
    fetcher = FakeFetcher({first: b"<article><img src='/one.jpg'><img src='/two.jpg'></article>",
                           "https://news.example.org/one.jpg": jpeg(5),
                           blocked: FetchFailure(blocked, "http_403", status=403),
                           third: b"<article><img src='/three.jpg'></article>", "https://other.example.org/three.jpg": jpeg(6)})
    report = collect_web(MemoryStore(tmp_path), path, target=3, fetcher=fetcher, max_per_source=1)
    assert report["status"] == "incomplete"
    assert report["total"] == 2
    assert report["host_errors"] == {"blocked.example.org": 1}
    assert "https://news.example.org/two.jpg" not in fetcher.calls
    assert fetcher.calls.count(blocked) == 1
