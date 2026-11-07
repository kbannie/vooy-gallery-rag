"""Download publicly accessible article photos discovered through Google search.

Search-result thumbnails are not used as originals. The article URL, image URL,
source identity labels, and any explicitly supplied licensing remain separate
from synthetic gallery metadata and generated visual captions.
"""

from __future__ import annotations

import hashlib
import io
import ipaddress
import json
import re
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import imagehash
from bs4 import BeautifulSoup
from PIL import Image, ImageOps, UnidentifiedImageError

from .metadata import extract_exif, synthetic_metadata

USER_AGENT = "VooyGalleryDemo/0.1 (public article photo retrieval; educational prototype)"
MAX_BYTES = 20 * 1024 * 1024
MEMBERS = {"RM", "Jin", "SUGA", "j-hope", "Jimin", "V", "Jung Kook"}
ALIASES = {"jungkook": "Jung Kook", "jung kook": "Jung Kook", "j-hope": "j-hope", "jhope": "j-hope",
           "suga": "SUGA", "rm": "RM", "jin": "Jin", "jimin": "Jimin", "v": "V"}
NON_PHOTO = re.compile(r"(?:^|[\W_])(logo|avatar|icon|emoji|banner|advertisement|advert|placeholder|"
                       r"loading|spacer|pixel|favicon|subscribe|newsletter)(?:$|[\W_])", re.IGNORECASE)
IMAGE_SUFFIX = re.compile(r"\.(?:jpe?g|png|webp)(?:$|\?)", re.IGNORECASE)


def public_url(value: str, page_url: str) -> str | None:
    """Normalize a public web URL without stripping functional/signed parameters."""
    if not value or value.strip().lower().startswith(("data:", "javascript:", "blob:")):
        return None
    parts = urlsplit(urljoin(page_url, value.strip()))
    host = (parts.hostname or "").lower()
    if parts.scheme not in {"http", "https"} or not host or parts.username or parts.password:
        return None
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        return None
    try:
        if not ipaddress.ip_address(host).is_global:
            return None
    except ValueError:
        pass
    # Preserve signed query bytes exactly, including escaping and parameter order.
    query = "&".join(part for part in parts.query.split("&")
                     if not unquote(part.split("=", 1)[0]).lower().startswith("utm_"))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))


def image_id(url: str) -> str:
    parts = urlsplit(url)
    stable_query = "&".join(part for part in parts.query.split("&")
                            if unquote(part.split("=", 1)[0]).lower() not in {"expires", "keyname", "signature"})
    stable_url = urlunsplit((parts.scheme, parts.netloc, parts.path, stable_query, ""))
    return "web-" + hashlib.sha256(stable_url.encode()).hexdigest()[:16]


def _largest_srcset(value: str) -> str | None:
    entries = re.findall(r"(\S+)\s+(\d+(?:\.\d+)?)([wx])(?:\s*,|\s*$)", value or "")
    if entries:
        return max(entries, key=lambda entry: float(entry[1]))[0]
    return (value or "").strip().split()[0] if (value or "").strip() else None


def discover_image_urls(html: str, page_url: str) -> list[dict[str, str]]:
    """Extract one best original URL per article image, with its visible alt text."""
    soup = BeautifulSoup(html, "html.parser")
    container = (soup.select_one(".entry-content") or soup.find("article") or soup.find("main")
                 or soup.select_one('[role="main"]') or soup.body)
    if container is None:
        return []
    for unwanted in container.select("script, style, nav, aside, footer, form, .sidebar, .advertisement, "
                                     ".ads, .ad, .related-posts, .related-articles, .author-bio, .comments"):
        unwanted.decompose()
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for image in container.find_all("img"):
        alt = " ".join(str(image.get("alt", "")).split())
        if NON_PHOTO.search(alt):
            continue
        anchor = image.find_parent("a", href=True)
        linked = str(anchor["href"]) if anchor and IMAGE_SUFFIX.search(str(anchor["href"])) else None
        candidates = [image.get("data-orig-file"), image.get("data-original"), linked,
                      _largest_srcset(str(image.get("data-srcset", ""))),
                      _largest_srcset(str(image.get("srcset", ""))), image.get("data-lazy-src"),
                      image.get("data-src"), image.get("src")]
        for candidate in candidates:
            if not candidate:
                continue
            url = public_url(str(candidate), page_url)
            if not url or NON_PHOTO.search(urlsplit(url).path) or urlsplit(url).path.lower().endswith((".gif", ".svg")):
                continue
            if re.fullmatch(r"(?:encrypted-)?tbn\d*\.(?:gstatic|google)\.com", urlsplit(url).hostname or ""):
                continue
            if url not in seen:
                seen.add(url)
                figure = image.find_parent("figure")
                caption = figure.find("figcaption") if figure else None
                result.append({"url": url, "alt": alt, "caption": caption.get_text(" ", strip=True) if caption else ""})
            break
    return result


class FetchFailure(Exception):
    def __init__(self, url: str, reason: str, status: int | None = None, retry_at: float | None = None):
        super().__init__(reason)
        self.url, self.reason, self.status, self.retry_at = url, reason, status, retry_at
        self.host = urlsplit(url).hostname or ""


class PublicRedirectHandler(HTTPRedirectHandler):
    def __init__(self, cooldowns: dict[str, float]):
        super().__init__()
        self.cooldowns = cooldowns

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        destination = public_url(newurl, req.full_url)
        if destination is None:
            raise FetchFailure(req.full_url, "non_public_redirect")
        host = urlsplit(destination).hostname or ""
        if self.cooldowns.get(host, 0) > time.time():
            raise FetchFailure(destination, "host_cooldown", status=429, retry_at=self.cooldowns[host])
        return super().redirect_request(req, fp, code, msg, headers, destination)


class WebFetcher:
    """Sequential requests with bounded bodies and independent per-host cooldowns."""

    def __init__(self, delay: float = 1.0, timeout: float = 15.0):
        self.delay = max(1.0, delay)
        self.timeout = min(15.0, max(1.0, timeout))
        self.last_request = 0.0
        self.cooldowns: dict[str, float] = {}
        self.rate_limits: list[dict[str, Any]] = []
        self.opener = build_opener(PublicRedirectHandler(self.cooldowns))

    def fetch(self, url: str, limit: int = MAX_BYTES) -> tuple[bytes, dict[str, str], str]:
        normalized = public_url(url, url)
        if normalized is None:
            raise FetchFailure(url, "not_a_public_url")
        host = urlsplit(normalized).hostname or ""
        if self.cooldowns.get(host, 0) > time.time():
            raise FetchFailure(normalized, "host_cooldown", status=429, retry_at=self.cooldowns[host])
        pause = self.delay - (time.monotonic() - self.last_request)
        if pause > 0:
            time.sleep(pause)
        self.last_request = time.monotonic()
        request = Request(normalized, headers={"User-Agent": USER_AGENT, "Accept": "text/html,image/*;q=0.9,*/*;q=0.5"})
        started = time.monotonic()
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                final_url = public_url(response.geturl(), normalized)
                if final_url is None:
                    raise FetchFailure(normalized, "non_public_redirect")
                headers = {key.lower(): value for key, value in response.headers.items()}
                content_length = headers.get("content-length", "")
                if content_length.isdigit() and int(content_length) > limit:
                    raise FetchFailure(normalized, "body_too_large")
                chunks: list[bytes] = []
                count = 0
                while True:
                    if time.monotonic() - started > self.timeout:
                        raise FetchFailure(normalized, "download_deadline")
                    chunk = response.read(min(64 * 1024, limit + 1 - count))
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > limit:
                        raise FetchFailure(normalized, "body_too_large")
                    chunks.append(chunk)
                return b"".join(chunks), headers, final_url
        except HTTPError as error:
            if error.code == 429:
                retry_after = error.headers.get("Retry-After", "")
                seconds = 60.0
                if retry_after.isdigit():
                    seconds = float(retry_after)
                elif retry_after:
                    try:
                        seconds = max(0, (parsedate_to_datetime(retry_after) - datetime.now(UTC)).total_seconds())
                    except (ValueError, TypeError):
                        pass
                blocked_host = urlsplit(error.geturl()).hostname or host
                retry_at = time.time() + max(1, seconds)
                self.cooldowns[blocked_host] = retry_at
                self.cooldowns[host] = retry_at
                self.rate_limits.append({"host": blocked_host, "retry_after": retry_after,
                                         "retry_after_seconds": seconds, "retry_at": datetime.fromtimestamp(retry_at, UTC).isoformat()})
                error.close()
                raise FetchFailure(normalized, "http_429", status=429, retry_at=retry_at) from error
            status = error.code
            error.close()
            raise FetchFailure(normalized, f"http_{status}", status=status) from error
        except (URLError, OSError, TimeoutError) as error:
            raise FetchFailure(normalized, f"network_error: {type(error).__name__}") from error


def collect_web(store: Any, sources_path: Path, target: int = 100, gallery_id: str = "bts-daily",
                progress: Callable[[dict[str, Any]], None] | None = None, *, max_per_source: int = 20,
                fetcher: WebFetcher | None = None) -> dict[str, Any]:
    """Collect daily-photo sources into Store, leaving unknown rights unspecified."""
    if target < 1 or max_per_source < 1:
        raise ValueError("target and max_per_source must be positive")
    sources = json.loads(Path(sources_path).read_text())
    if not isinstance(sources, list) or any(not isinstance(source, dict) or not source.get("page_url") for source in sources):
        raise ValueError("sources must be a JSON list of objects with page_url")
    existing = store.list_photos(gallery_id)
    active = [photo for photo in existing if photo.get("collection_theme") == "daily"
              and photo.get("index_status") != "excluded" and photo.get("review", {}).get("included") is not False]
    known_ids = {photo["photo_id"] for photo in existing}
    hashes = {photo["sha256"] for photo in existing if photo.get("sha256")}
    phashes = [imagehash.hex_to_hash(photo["phash"]) for photo in existing if photo.get("phash")]
    per_source = Counter(photo.get("source", {}).get("page_url") for photo in active)
    stats: dict[str, Any] = {"status": "incomplete", "target": target, "existing": len(active), "downloaded": 0,
                             "total": len(active), "sources_processed": 0, "skipped": {}, "errors": [],
                             "host_errors": {}, "rate_limits": [], "visual_review_required": True}
    if stats["total"] >= target:
        stats["status"] = "target_reached"
        return stats
    skipped: Counter[str] = Counter()
    host_errors: Counter[str] = Counter()
    fetcher = fetcher or WebFetcher()
    images, thumbnails = store.data_dir / "images", store.data_dir / "thumbnails"
    images.mkdir(parents=True, exist_ok=True)
    thumbnails.mkdir(parents=True, exist_ok=True)

    def failure(error: FetchFailure) -> None:
        skipped[error.reason.split(":", 1)[0]] += 1
        host_errors[error.host] += 1
        detail = {"url": error.url, "host": error.host, "reason": error.reason, "status": error.status}
        if error.retry_at:
            detail["retry_at"] = datetime.fromtimestamp(error.retry_at, UTC).isoformat()
        stats["errors"].append(detail)
        if progress:
            progress({"event": "source_error", **detail})

    for source in sources:
        if stats["total"] >= target:
            break
        page_url = str(source["page_url"])
        if per_source[page_url] >= max_per_source:
            skipped["per_source_limit"] += 1
            continue
        stats["sources_processed"] += 1
        try:
            html, _, resolved_page = fetcher.fetch(page_url, limit=5 * 1024 * 1024)
            candidates = discover_image_urls(html.decode("utf-8", errors="replace"), resolved_page)
        except FetchFailure as error:
            failure(error)
            continue
        if not candidates:
            skipped["no_article_images"] += 1
        for candidate in candidates:
            if stats["total"] >= target or per_source[page_url] >= max_per_source:
                break
            image_url = candidate["url"]
            photo_id = image_id(image_url)
            if photo_id in known_ids:
                skipped["already_collected_or_excluded"] += 1
                continue
            try:
                payload, _, resolved_image = fetcher.fetch(image_url)
                sha = hashlib.sha256(payload).hexdigest()
                if sha in hashes:
                    skipped["exact_duplicate"] += 1
                    continue
                with Image.open(io.BytesIO(payload)) as original:
                    original.load()
                    if original.format not in {"JPEG", "PNG", "WEBP"} or getattr(original, "is_animated", False):
                        skipped["unsupported_format"] += 1
                        continue
                    if min(original.size) < 350:
                        skipped["too_small"] += 1
                        continue
                    width, height = original.size
                    image = ImageOps.exif_transpose(original).convert("RGB")
                    fingerprint = imagehash.phash(image)
                    if any(fingerprint - previous <= 5 for previous in phashes):
                        skipped["near_duplicate"] += 1
                        continue
                    extension, mime = {"JPEG": (".jpg", "image/jpeg"), "PNG": (".png", "image/png"),
                                       "WEBP": (".webp", "image/webp")}[original.format]
                    image_path, thumbnail_path = images / f"{photo_id}{extension}", thumbnails / f"{photo_id}.jpg"
                    temporary = image_path.with_suffix(extension + ".part")
                    temporary.write_bytes(payload)
                    temporary.replace(image_path)
                    image.thumbnail((640, 640))
                    image.save(thumbnail_path, "JPEG", quality=85)
                title = str(source.get("title") or "BTS daily photo article")
                author = str(source.get("author") or "Not specified")
                license_name = str(source.get("license_name") or "Not specified")
                license_url = str(source.get("license_url") or "")
                article_identities = list(dict.fromkeys(ALIASES.get(str(name).lower(), str(name)) for name in source.get("identities", [])))
                article_identities = [name for name in article_identities if name in MEMBERS]
                image_context = " ".join([candidate["alt"], candidate.get("caption", "")])
                patterns = {"RM": r"\bRM\b", "Jin": r"\bJin\b(?!\s+Air)", "SUGA": r"\bSuga\b",
                            "j-hope": r"\bj[- _]?hope\b", "Jimin": r"\bJimin\b", "Jung Kook": r"\bJung[- _]?kook\b"}
                identities = [name for name, pattern in patterns.items() if re.search(pattern, image_context, re.IGNORECASE)]
                if re.search(r"\bV\b", image_context) and not re.search(r"\bVol\.?\s+V\b", image_context):
                    identities.append("V")
                identity_source = "image_alt_or_caption"
                if not identities:
                    identities = article_identities if len(article_identities) == 1 else []
                    identity_source = "article_context" if identities else "unspecified"
                photo = {"photo_id": photo_id, "gallery_id": gallery_id, "collection_theme": "daily",
                         "image_path": str(image_path.relative_to(store.data_dir)),
                         "thumbnail_path": str(thumbnail_path.relative_to(store.data_dir)),
                         "sha256": sha, "phash": str(fingerprint), "mime_type": mime, "width": width, "height": height,
                         "identities": identities, "identity_source": identity_source,
                         "metadata": synthetic_metadata(photo_id), "raw_exif": extract_exif(payload),
                         "source": {"page_url": page_url, "resolved_page_url": resolved_page, "original_url": resolved_image,
                                    "discovered_image_url": image_url, "title": title, "description": image_context,
                                    "article_identities": article_identities,
                                    "author": author, "license_name": license_name, "license_url": license_url,
                                    "attribution": f"{title} — {author} — License: {license_name}. {page_url} {license_url}".strip(),
                                    "categories": ["BTS", "daily", "casual", "travel"],
                                    "discovery": "Google/web image search"}}
                store.upsert_photo(photo)
                known_ids.add(photo_id)
                hashes.add(sha)
                phashes.append(fingerprint)
                per_source[page_url] += 1
                stats["downloaded"] += 1
                stats["total"] += 1
                if progress:
                    progress({"event": "downloaded", "photo_id": photo_id, "total": stats["total"], "target": target})
            except FetchFailure as error:
                failure(error)
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
                skipped["decode_or_storage_error"] += 1
                stats["errors"].append({"url": image_url, "host": urlsplit(image_url).hostname, "reason": str(error)[:300]})
    stats["status"] = "target_reached" if stats["total"] >= target else "incomplete"
    stats["skipped"] = dict(skipped)
    stats["host_errors"] = dict(host_errors)
    stats["rate_limits"] = fetcher.rate_limits
    return stats
