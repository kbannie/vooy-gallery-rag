"""A bounded, resumable Wikimedia Commons photo collector.

Uses the public categorymembers and imageinfo APIs; it does not scrape search
engine thumbnails or treat a Wikimedia upload time as a capture time.
"""

from __future__ import annotations

import hashlib
import html
import io
import re
import time
from collections import Counter, deque
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlparse, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import httpx
import imagehash
from PIL import Image, ImageOps, UnidentifiedImageError

from .metadata import extract_exif, synthetic_metadata

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "VooyGalleryDemo/0.1 (BTS educational photo retrieval; Wikimedia Commons API)"
DEFAULT_CATEGORIES = (
    "Category:BTS by year",
    "Category:Jung Kook by year",
    "Category:Jimin by year",
    "Category:V (vocalist) by year",
    "Category:RM by year",
    "Category:Jin (vocalist) by year",
    "Category:Suga by year",
    "Category:J-Hope by year",
    "Category:BTS World Tour: Love Yourself",
)
ALLOWED_MIME = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
REJECT_TITLE = re.compile(
    r"\b(logo|logotype|poster|album cover|cover art|merchandise|lightstick|light stick|"
    r"billboard advertisement|advertisement|fans only|audience|crowd|cosplay|drawing|"
    r"painting|illustration|wax figure|statue|mural|graffiti|signature|autograph|"
    r"photocard|photo card|fanart|fan art)\b", re.IGNORECASE
)
MEMBERS = {
    "RM": (r"\brm\b", r"\brap monster\b", r"\bkim nam[- ]?joon\b", "남준"),
    "Jin": (r"\bjin\b(?!\s+air\b)", r"\bkim seok[- ]?jin\b", "석진"),
    "SUGA": (r"\bsuga\b", r"\bagust d\b", r"\bmin yoon[- ]?gi\b", "슈가", "윤기"),
    "j-hope": (r"\bj[- ]?hope\b", r"\bjung ho[- ]?seok\b", "제이홉", "호석"),
    "Jimin": (r"\bjimin\b", r"\bpark ji[- ]?min\b", "지민"),
    "V": (r"\bv \(vocalist\)", r"\bkim tae[- ]?hyung\b", r"\btaehyung\b", "태형"),
    "Jung Kook": (r"\bjung[- ]?kook\b", r"\bjeon jeong[- ]?guk\b", "정국"),
}


def clean_html(value: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]*>", " ", value or "")).split())


def source_identities(title: str, description: str, categories: list[str]) -> list[str]:
    """Use explicit source labels, never visual identity recognition."""
    text = " ".join([title, description, *categories])
    found = [member for member, patterns in MEMBERS.items() if any(re.search(p, text, re.IGNORECASE) for p in patterns)]
    # A standalone V must be an explicit BTS-person label, not a Roman numeral
    # elsewhere in a concert title or a sentence about the group.
    v_label = r"\bBTS(?:['’]s)?[\s_-]+V(?=[\s_.-]|$)|\bV[\s_-]+(?:of[\s_-]+)?BTS\b"
    if "V" not in found and re.search(v_label, title):
        found.append("V")
    return found


def accepted_license(name: str, url: str, license_code: str = "") -> bool:
    text = f"{name} {license_code}".lower().replace("_", "-")
    if any(term in text for term in ("noncommercial", "non-commercial", "no derivatives", "by-nc", "by-nd")):
        return False
    if "public domain" in text or "cc0" in text:
        return True
    if re.search(r"\bcc[ -]by(?:[ -]sa)?(?:[ -]\d|\b)", text):
        return True
    return bool(re.match(r"https?://creativecommons\.org/(?:licenses/by(?:-sa)?/|publicdomain/(?:zero|mark)/)", url))


class UrllibTransport(httpx.BaseTransport):
    """Use the platform's standard-library HTTP stack for Commons compatibility.

    Some network environments reject HTTPX's direct stack while allowing the
    same public API request through urllib. The explicit application User-Agent
    and normal HTTP status handling are retained; no browser impersonation.
    """

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        upstream = Request(str(request.url), headers={"User-Agent": request.headers.get("User-Agent", USER_AGENT)})
        try:
            timeout = request.extensions.get("timeout", {}).get("read", 45)
            with urlopen(upstream, timeout=timeout) as response:
                return httpx.Response(response.status, headers=list(response.headers.items()), content=response.read(), request=request)
        except HTTPError as error:
            with error:
                return httpx.Response(error.code, headers=list(error.headers.items()), content=error.read(), request=request)
        except (URLError, TimeoutError, OSError) as error:
            raise httpx.NetworkError(str(error), request=request) from error


class CommonsClient:
    def __init__(self, client: httpx.Client | None = None, delay: float = 1.0, retries: int = 2,
                 progress: Callable[[dict[str, Any]], None] | None = None):
        self.client = client or httpx.Client(transport=UrllibTransport(), timeout=45, follow_redirects=True,
                                            headers={"User-Agent": USER_AGENT})
        self.owns_client = client is None
        self.delay = max(delay, 0)
        self.retries = retries
        self.progress = progress
        self.telemetry: dict[str, Any] = {"requests": 0, "rate_limits": []}
        self._last_request = 0.0

    def close(self) -> None:
        if self.owns_client:
            self.client.close()

    def _wait(self) -> None:
        remaining = self.delay - (time.monotonic() - self._last_request)
        if remaining > 0:
            time.sleep(remaining)
        self._last_request = time.monotonic()

    def request(self, url: str, *, retry_limit: int | None = None, **kwargs: Any) -> httpx.Response:
        last_error: Exception | None = None
        retries = self.retries if retry_limit is None else retry_limit
        for attempt in range(retries + 1):
            self._wait()
            started = time.monotonic()
            try:
                response = self.client.get(url, headers={"User-Agent": USER_AGENT}, **kwargs)
                self.telemetry["requests"] += 1
                if self.progress:
                    self.progress({"event": "http_response", "host": urlparse(url).hostname,
                                   "status": response.status_code, "elapsed_seconds": round(time.monotonic() - started, 2)})
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                response.raise_for_status()
                return response
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as error:
                if isinstance(error, httpx.HTTPStatusError) and error.response.status_code < 500 and error.response.status_code != 429:
                    raise
                last_error = error
                retry_after = error.response.headers.get("Retry-After", "") if isinstance(error, httpx.HTTPStatusError) else ""
                pause = min(2 ** attempt, 16)
                if retry_after.isdigit():
                    pause = float(retry_after)
                elif retry_after:
                    try:
                        pause = max(0, (parsedate_to_datetime(retry_after) - datetime.now(UTC)).total_seconds())
                    except (ValueError, TypeError):
                        pass
                if isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 429:
                    pause = max(pause, 10)
                    self.telemetry["rate_limits"].append({"host": urlparse(url).hostname,
                                                          "retry_after": retry_after,
                                                          "pause_seconds": pause})
                if self.progress:
                    self.progress({"event": "request_retry", "host": urlparse(url).hostname,
                                   "attempt": attempt + 1, "pause_seconds": pause,
                                   "elapsed_seconds": round(time.monotonic() - started, 2), "error": str(error)[:180]})
                # Even when abandoning this file, wait out the rate-limit window
                # before proceeding to another image on the same public service.
                if attempt < retries or (isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 429):
                    time.sleep(pause)
                if attempt == retries:
                    raise
        raise RuntimeError("Request failed") from last_error

    def api(self, **params: Any) -> dict[str, Any]:
        base = {"action": "query", "format": "json", "formatversion": 2, "maxlag": 5, **params}
        for attempt in range(self.retries + 1):
            payload = self.request(API_URL, params=base, timeout=30).json()
            error = payload.get("error")
            if not error:
                return payload
            if error.get("code") in {"maxlag", "ratelimited"} and attempt < self.retries:
                time.sleep(min(2 ** (attempt + 1), 30))
                continue
            raise RuntimeError(f"Commons API: {error.get('code')}: {error.get('info', '')}")
        raise RuntimeError("Commons API maxlag retry budget exhausted")

    def category(self, title: str) -> Iterator[dict[str, Any]]:
        continuation: dict[str, Any] = {}
        while True:
            data = self.api(list="categorymembers", cmtitle=title, cmtype="file|subcat", cmlimit=500, **continuation)
            yield from data.get("query", {}).get("categorymembers", [])
            continuation = data.get("continue", {})
            if not continuation:
                break

    def image_info(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payload = self.api(pageids="|".join(str(p["pageid"]) for p in candidates), prop="imageinfo|categories",
                           iiprop="url|size|mime|sha1|extmetadata", cllimit=500)
        pages = payload.get("query", {}).get("pages", [])
        if isinstance(pages, dict):
            pages = list(pages.values())
        # The traversal category is retained even if page category pagination is omitted.
        traversal = {p["pageid"]: p.get("traversal_categories", []) for p in candidates}
        for page in pages:
            categories = [c["title"] for c in page.get("categories", [])]
            page["source_categories"] = sorted(set(categories + traversal.get(page["pageid"], [])))
        return pages


def iter_candidates(api: CommonsClient, roots: tuple[str, ...], max_categories: int, max_candidates: int) -> Iterator[dict[str, Any]]:
    """Round-robin per-root walks keep group and member photos represented."""
    seen_categories: set[str] = set()
    seen_files: set[int] = set()

    def walk(root: str) -> Iterator[dict[str, Any]]:
        queue = deque([(root, 0)])
        while queue and len(seen_categories) < max_categories:
            category, depth = queue.popleft()
            if category in seen_categories:
                continue
            seen_categories.add(category)
            entries = list(api.category(category))
            # Hash sorting avoids taking only alphabetically first events/filenames.
            entries.sort(key=lambda item: hashlib.sha256(item["title"].encode()).hexdigest())
            for item in entries:
                if item.get("ns") == 14:
                    if depth < 4 and not REJECT_TITLE.search(item["title"]):
                        queue.append((item["title"], depth + 1))
                elif item.get("ns") == 6 and item["pageid"] not in seen_files:
                    seen_files.add(item["pageid"])
                    if not REJECT_TITLE.search(item["title"]):
                        yield {**item, "traversal_categories": [category, root]}

    streams = deque(walk(root) for root in roots)
    count = 0
    while streams and count < max_candidates:
        stream = streams.popleft()
        try:
            candidate = next(stream)
        except StopIteration:
            continue
        streams.append(stream)
        count += 1
        yield candidate


def _source(page: dict[str, Any]) -> dict[str, Any] | None:
    if not page.get("imageinfo"):
        return None
    info = page["imageinfo"][0]
    ext = info.get("extmetadata", {})

    def field(name: str) -> str:
        return clean_html(str(ext.get(name, {}).get("value", "")))

    license_name, license_url = field("LicenseShortName"), field("LicenseUrl")
    if not accepted_license(license_name, license_url, field("License")):
        return None
    author = field("Artist") or field("Attribution")
    if not author and "public domain" not in license_name.lower() and "cc0" not in license_name.lower():
        return None
    page_url = info.get("descriptionurl", "")
    title = page.get("title", "")
    # Analytics query strings make an otherwise immutable public file miss CDN
    # caches. Keep the exact API-returned URL as provenance, request its canonical
    # image URL without only the known utm_* analytics parameters.
    original_url = info.get("url", "")
    parts = urlsplit(original_url)
    query = urlencode([(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True) if not key.startswith("utm_")])
    canonical_url = urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))
    return {
        "page_url": page_url,
        "original_url": canonical_url,
        "api_original_url": original_url,
        "title": title,
        "description": field("ImageDescription"),
        "author": author or "Unknown author",
        "license_name": license_name,
        "license_url": license_url,
        "attribution": f"{title} — {author or 'Unknown author'} — {license_name}. {page_url} {license_url}".strip(),
        "categories": page.get("source_categories", []),
        "commons_page_id": page["pageid"],
        "raw_extmetadata": ext,
    }


def collect(store: Any, target: int = 100, gallery_id: str = "bts-demo", *, max_candidates: int = 2000,
            max_categories: int = 160, delay: float = 1.0, progress: Callable[[dict[str, Any]], None] | None = None,
            client: httpx.Client | None = None, roots: tuple[str, ...] = DEFAULT_CATEGORIES,
            seed: int = 42, min_dimension: int = 480, phash_distance: int = 5) -> dict[str, Any]:
    """Collect up to target usable photos; return incomplete status on source shortage.

    Structural/source checks are automated. A final visual contact-sheet review is
    still needed to catch incidental crowd photos or materially different crops.
    """
    if target < 1 or max_candidates < 1 or max_categories < 1:
        raise ValueError("target, max_candidates and max_categories must be positive")
    existing = store.list_photos(gallery_id)
    excluded_ids = {photo["photo_id"] for photo in existing if photo.get("index_status") == "excluded"}
    # PostgreSQL owns the original and preview bytes. Local cache eviction must
    # not make an existing database photograph look absent or trigger a download.
    known_ids = {photo["photo_id"] for photo in existing}
    active_ids = known_ids - excluded_ids
    known_hashes = {photo["sha256"] for photo in existing if photo["photo_id"] in known_ids}
    known_phashes = [imagehash.hex_to_hash(photo["phash"]) for photo in existing if photo.get("phash") and photo["photo_id"] in known_ids]
    stats: dict[str, Any] = {"target": target, "existing": len(active_ids), "excluded": len(excluded_ids),
                             "downloaded": 0, "candidates": 0, "total": len(active_ids),
                             "status": "incomplete", "skipped": {}, "errors": [],
                             "visual_review_required": True, "network": {"requests": 0, "rate_limits": []}}
    if len(active_ids) >= target:
        stats["status"] = "target_reached"
        return stats
    image_dir, thumb_dir = store.data_dir / "images", store.data_dir / "thumbnails"
    image_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    skipped: Counter[str] = Counter()
    api = CommonsClient(client, delay=delay, progress=progress)
    pending: list[dict[str, Any]] = []

    def process(batch: list[dict[str, Any]]) -> None:
        pages = api.image_info(batch)
        for page in pages:
            if stats["total"] >= target:
                return
            photo_id = f"commons-{page['pageid']}"
            if photo_id in known_ids:
                skipped["already_excluded" if photo_id in excluded_ids else "already_collected"] += 1
                continue
            info = (page.get("imageinfo") or [{}])[0]
            mime = info.get("mime")
            if mime not in ALLOWED_MIME or min(info.get("width", 0), info.get("height", 0)) < min_dimension:
                skipped["format_or_dimensions"] += 1
                continue
            if info.get("size", 0) > 20 * 1024 * 1024:
                skipped["too_large"] += 1
                continue
            source = _source(page)
            if source is None:
                skipped["license_or_author"] += 1
                continue
            if REJECT_TITLE.search(source["title"]):
                skipped["non_photo_subject"] += 1
                continue
            host = urlparse(source["original_url"]).hostname
            if host != "upload.wikimedia.org":
                skipped["unexpected_image_host"] += 1
                continue
            try:
                payload = api.request(source["original_url"], timeout=15, retry_limit=1).content
                if len(payload) > 20 * 1024 * 1024:
                    skipped["too_large"] += 1
                    continue
                sha = hashlib.sha256(payload).hexdigest()
                if sha in known_hashes:
                    skipped["exact_duplicate"] += 1
                    continue
                with Image.open(io.BytesIO(payload)) as raw:
                    raw.load()
                    if raw.format not in {"JPEG", "PNG", "WEBP"} or getattr(raw, "is_animated", False):
                        skipped["unsupported_actual_format"] += 1
                        continue
                    image = ImageOps.exif_transpose(raw).convert("RGB")
                    if min(image.size) < min_dimension:
                        skipped["format_or_dimensions"] += 1
                        continue
                    fingerprint = imagehash.phash(image)
                    if any(fingerprint - previous <= phash_distance for previous in known_phashes):
                        skipped["near_duplicate"] += 1
                        continue
                    image_path = image_dir / f"{photo_id}{ALLOWED_MIME[mime]}"
                    thumb_path = thumb_dir / f"{photo_id}.jpg"
                    # Original bytes are stored unchanged; only the preview is resized.
                    temp_path = image_path.with_suffix(image_path.suffix + ".part")
                    temp_path.write_bytes(payload)
                    temp_path.replace(image_path)
                    image.thumbnail((640, 640))
                    image.save(thumb_path, "JPEG", quality=85)
                photo = {
                    "photo_id": photo_id, "gallery_id": gallery_id,
                    "image_path": str(image_path.relative_to(store.data_dir)),
                    "thumbnail_path": str(thumb_path.relative_to(store.data_dir)),
                    "sha256": sha, "phash": str(fingerprint), "mime_type": mime,
                    "width": info["width"], "height": info["height"], "source": source,
                    "identities": source_identities(source["title"], source["description"], source["categories"]),
                    "metadata": synthetic_metadata(photo_id, seed), "raw_exif": extract_exif(payload),
                }
                store.upsert_photo(photo)
                known_ids.add(photo_id)
                known_hashes.add(sha)
                known_phashes.append(fingerprint)
                stats["downloaded"] += 1
                stats["total"] += 1
                if progress:
                    progress({"event": "downloaded", "photo_id": photo_id, "total": stats["total"], "target": target})
            except (httpx.HTTPError, UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
                skipped["download_or_decode_error"] += 1
                stats["errors"].append({"photo_id": photo_id, "error": str(error)[:300]})

    try:
        for candidate in iter_candidates(api, roots, max_categories, max_candidates):
            stats["candidates"] += 1
            candidate_id = f"commons-{candidate['pageid']}"
            if candidate_id in known_ids:
                skipped["already_excluded" if candidate_id in excluded_ids else "already_collected"] += 1
                continue
            pending.append(candidate)
            if len(pending) >= 10:
                process(pending)
                pending.clear()
            if stats["total"] >= target:
                break
        if pending and stats["total"] < target:
            process(pending)
    finally:
        stats["network"] = api.telemetry
        api.close()
    stats["status"] = "target_reached" if stats["total"] >= target else "incomplete"
    stats["skipped"] = dict(skipped)
    return stats
