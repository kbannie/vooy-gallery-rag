"""Resumable indexing and exact metadata filters followed by semantic retrieval."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from threading import Event
from typing import Any
from zoneinfo import ZoneInfo

from openai import APIError, AuthenticationError, BadRequestError, NotFoundError, PermissionDeniedError

from .ai import (
    Caption,
    ConfigurationError,
    GalleryAI,
    QueryPlan,
    Ranking,
    caption_key,
    embedding_key,
    embedding_text,
    normalized_vector,
)

KST = ZoneInfo("Asia/Seoul")
IDENTITY_ALIASES = {
    "rm": "RM", "알엠": "RM", "남준": "RM", "김남준": "RM",
    "jin": "Jin", "진": "Jin", "석진": "Jin", "김석진": "Jin",
    "suga": "SUGA", "슈가": "SUGA", "윤기": "SUGA", "민윤기": "SUGA",
    "j-hope": "j-hope", "jhope": "j-hope", "제이홉": "j-hope", "호석": "j-hope",
    "jimin": "Jimin", "지민": "Jimin", "박지민": "Jimin",
    "v": "V", "뷔": "V", "태형": "V", "김태형": "V",
    "jung kook": "Jung Kook", "jungkook": "Jung Kook", "정국": "Jung Kook", "전정국": "Jung Kook",
}
LOCATION_ALIASES = {
    "서울": "서울", "서울시": "서울", "서울특별시": "서울", "seoul": "서울",
    "부산": "부산", "부산시": "부산", "부산광역시": "부산", "busan": "부산",
    "제주": "제주", "제주도": "제주", "제주특별자치도": "제주", "jeju": "제주",
    "인천": "인천", "인천시": "인천", "인천광역시": "인천", "incheon": "인천",
}


def _ai(store: Any, settings: Any, client: Any) -> Any:
    if client is not None and hasattr(client, "plan_query") and hasattr(client, "embed"):
        return client
    return GalleryAI(store, settings, client=client)


def index_gallery(store: Any, settings: Any, limit: int | None = None, client: Any = None,
                  workers: int = 1, progress: Callable[[dict], None] | None = None) -> dict:
    """Index the first ``limit`` photos deterministically, reusing valid work on restart."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    if not isinstance(workers, int) or isinstance(workers, bool) or not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    photos = store.list_photos(settings.gallery_id)
    if limit is not None:
        photos = photos[:limit]
    if not photos:
        raise ConfigurationError("사진이 없습니다. 먼저 collect를 실행하세요.")
    service = _ai(store, settings, client)
    stats = {
        "status": "ok", "requested": len(photos), "indexed": 0, "reused": 0,
        "excluded": 0, "failed": [], "gallery_id": settings.gallery_id,
    }
    before = store.usage_summary()
    stopped = Event()

    def process_photo(photo: dict) -> dict:
        if stopped.is_set():
            return {"event": "cancelled", "photo_id": photo["photo_id"]}
        if photo.get("review", {}).get("included") is False:
            photo["index_status"] = "excluded"
            photo["exclusion_reason"] = photo["review"].get("reason", "사진 검수에서 제외됨")
            store.upsert_photo(photo)
            return {"event": "excluded", "photo_id": photo["photo_id"]}
        try:
            expected_caption_key = caption_key(photo, settings)
            reused_caption = (
                photo.get("caption_cache_key") == expected_caption_key and photo.get("caption")
            )
            if not reused_caption:
                if stopped.is_set():
                    return {"event": "cancelled", "photo_id": photo["photo_id"]}
                caption = Caption.model_validate(service.caption(photo)).model_dump()
                photo["caption"] = caption
                photo["caption_cache_key"] = expected_caption_key
                photo["index_status"] = "captioned"
                store.upsert_photo(photo)
            else:
                caption = Caption.model_validate(photo["caption"]).model_dump()
            if not caption["is_photo"] or not caption["contains_people"]:
                photo["index_status"] = "excluded"
                photo["exclusion_reason"] = "Vision 검수: 실제 인물이 등장하는 사진이 아님"
                store.upsert_photo(photo)
                return {"event": "excluded", "photo_id": photo["photo_id"]}
            key = embedding_key(photo, settings)
            vector = store.get_embedding(photo["photo_id"], cache_key=key)
            vector_reused = vector is not None
            if vector is not None:
                normalized_vector(vector, settings.embedding_dimensions)
            else:
                if stopped.is_set():
                    return {"event": "cancelled", "photo_id": photo["photo_id"]}
                vector = service.embed(embedding_text(photo))
                vector = normalized_vector(vector, settings.embedding_dimensions)
                store.set_embedding(photo["photo_id"], vector, key, settings.embedding_model)
            photo["index_status"] = "indexed"
            photo["embedding_cache_key"] = key
            photo["embedding_model"] = settings.embedding_model
            photo["embedding_dimensions"] = settings.embedding_dimensions
            photo.pop("exclusion_reason", None)
            store.upsert_photo(photo)
            return {"event": "indexed", "photo_id": photo["photo_id"], "reused": vector_reused}
        except (ConfigurationError, AuthenticationError, PermissionDeniedError, NotFoundError, BadRequestError):
            # Configuration, inaccessible models, invalid API requests: do not retry on 100 files.
            stopped.set()
            raise
        except (APIError, OSError, RuntimeError, ValueError) as exc:
            return {"event": "failed", "photo_id": photo["photo_id"], "error": str(exc)}

    processed = 0

    def record(result: dict) -> None:
        nonlocal processed
        event = result["event"]
        if event == "cancelled":
            return
        processed += 1
        if event == "failed":
            stats["failed"].append({"photo_id": result["photo_id"], "error": result["error"]})
            stats["status"] = "partial"
        else:
            stats[event] += 1
            if result.get("reused"):
                stats["reused"] += 1
        if progress:
            progress({**result, "processed": processed, "total": len(photos),
                      "indexed": stats["indexed"], "excluded": stats["excluded"],
                      "failed": len(stats["failed"]), "reused_count": stats["reused"]})

    if workers == 1:
        for photo in photos:
            record(process_photo(photo))
    else:
        # Keep only workers tasks in flight, so a fatal response cannot leave an
        # entire gallery queued for billable API calls. Store connections are per call.
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gallery-index")
        remaining = iter(photos)
        pending = set()
        try:
            for _ in range(min(workers, len(photos))):
                pending.add(executor.submit(process_photo, next(remaining)))
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    record(future.result())
                if not stopped.is_set():
                    for _ in completed:
                        photo = next(remaining, None)
                        if photo is not None:
                            pending.add(executor.submit(process_photo, photo))
        finally:
            stopped.set()
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
    order = {photo["photo_id"]: i for i, photo in enumerate(photos)}
    stats["failed"].sort(key=lambda item: order[item["photo_id"]])
    # An intentional suitability exclusion is not a failed indexing operation.
    # Collector top-ups use the explicit excluded/indexed counts independently.
    stats["usage_before"] = before
    stats["usage_after"] = store.usage_summary()
    return stats


def canonical_identity(value: str) -> str:
    return IDENTITY_ALIASES.get(value.strip().lower(), value.strip())


def canonical_location(value: str) -> str:
    return LOCATION_ALIASES.get(value.strip().lower(), value.strip())


def parse_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        # Do not silently assign a timezone to real EXIF values with unknown offset.
        raise ValueError("촬영 시각 조건에는 시간대가 필요합니다.")
    return result


def matches_filters(photo: dict, plan: QueryPlan) -> bool:
    metadata = photo.get("metadata") or {}
    if plan.location:
        location = metadata.get("location")
        if not location or canonical_location(location) != canonical_location(plan.location):
            return False
    if plan.people:
        identities = {canonical_identity(name) for name in photo.get("identities", [])}
        if not {canonical_identity(name) for name in plan.people}.issubset(identities):
            return False
    if plan.captured_from or plan.captured_until:
        try:
            captured = parse_time(metadata.get("captured_at"))
        except (ValueError, TypeError):
            return False
        if captured is None:
            return False
        start, end = parse_time(plan.captured_from), parse_time(plan.captured_until)
        if start is not None and captured < start:
            return False
        if end is not None and captured >= end:
            return False
    return True


class SearchService:
    def __init__(self, store: Any, settings: Any, client: Any = None):
        self.store = store
        self.settings = settings
        self.ai = _ai(store, settings, client)

    def search(
        self,
        query: str,
        gallery_id: str | None = None,
        limit: int = 3,
        now: datetime | None = None,
    ) -> dict:
        started = time.monotonic()
        gallery_id = gallery_id or self.settings.gallery_id
        if gallery_id != self.settings.gallery_id:
            raise ValueError("This local demo only serves its configured gallery.")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 3:
            raise ValueError("limit must be between 1 and 3")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("검색어를 입력하세요.")
        query = query.strip()
        if len(query) > 4000:
            raise ValueError("검색어는 4,000자 이내로 입력하세요.")
        now = now or datetime.now(KST)
        if now.tzinfo is None:
            raise ValueError("now must include its timezone")
        now = now.astimezone(KST)
        photos = self.store.list_photos(gallery_id)
        if not photos:
            raise ConfigurationError("사진이 없습니다. 먼저 collect와 index를 실행하세요.")
        eligible_cache_keys = {}
        for photo in photos:
            if photo.get("index_status") != "indexed" or not photo.get("caption"):
                continue
            # Build a metadata-only manifest. PostgreSQL checks the embedding cache keys
            # and performs all vector ranking; no corpus vectors leave the database.
            if photo.get("caption_cache_key") != caption_key(photo, self.settings):
                continue
            eligible_cache_keys[photo["photo_id"]] = embedding_key(photo, self.settings)
        indexed_count = self.store.count_indexed(gallery_id, eligible_cache_keys)
        if not indexed_count:
            raise ConfigurationError("현재 모델 설정으로 인덱싱된 사진이 없습니다. index를 실행하세요.")
        locations = sorted({
            photo.get("metadata", {}).get("location")
            for photo in photos
            if photo["photo_id"] in eligible_cache_keys
            if photo.get("metadata", {}).get("location")
        })
        plan = QueryPlan.model_validate(self.ai.plan_query(query, now, locations))
        base = {
            "status": "ok", "query": query, "gallery_id": gallery_id, "results": [],
            "requested_limit": limit, "now": now.isoformat(), "clarification": None,
            "applied_filters": {
                "location": canonical_location(plan.location) if plan.location else None,
                "captured_from": plan.captured_from,
                "captured_until": plan.captured_until,
                "people": [canonical_identity(name) for name in plan.people],
            },
            "semantic_query": plan.semantic_query,
            "indexed_count": indexed_count,
        }

        def finish() -> dict:
            base["elapsed_seconds"] = round(time.monotonic() - started, 4)
            return self.store.save_search(base)

        if plan.status == "needs_clarification":
            base["status"] = "needs_clarification"
            base["clarification"] = plan.clarification or "어떤 장소나 시점의 사진인지 알려주세요."
            return finish()
        try:
            start, end = parse_time(plan.captured_from), parse_time(plan.captured_until)
            if start is not None and end is not None and start >= end:
                raise ValueError("invalid range")
        except (TypeError, ValueError):
            base["status"] = "needs_clarification"
            base["clarification"] = "찾을 사진의 날짜 범위를 다시 알려주세요."
            return finish()
        # Check hard filters in SQL before spending tokens embedding a query. The same
        # filters are applied again inside the pgvector ORDER BY ... <=> ... query.
        metadata_rows, filtered_count = self.store.search_candidates(
            None, gallery_id, base["applied_filters"], eligible_cache_keys,
            limit=limit if not plan.semantic_query.strip() else 1,
        )
        base["filtered_count"] = filtered_count
        if not filtered_count:
            base["status"] = "no_match"
            return finish()
        chosen: list[tuple[dict, float | None, str]] = []
        if not plan.semantic_query.strip():
            chosen = [
                (row["photo"], None, "요청한 메타데이터 조건과 일치합니다.")
                for row in metadata_rows[:limit]
            ]
        else:
            query_vector = normalized_vector(self.ai.embed(plan.semantic_query), self.settings.embedding_dimensions)
            rows, _ = self.store.search_candidates(
                query_vector.tolist(), gallery_id, base["applied_filters"], eligible_cache_keys, limit=10
            )
            candidates = [(row["photo"], row["score"]) for row in rows]
            candidate_map = {photo["photo_id"]: (photo, score) for photo, score in candidates}
            ranking = Ranking.model_validate(self.ai.rerank(query, plan.semantic_query, [p for p, _ in candidates]))
            seen: set[str] = set()
            for item in ranking.photos:
                # Models can still invent IDs or duplicate them despite schema validation.
                if item.photo_id not in candidate_map or item.photo_id in seen:
                    continue
                seen.add(item.photo_id)
                if item.relevant:
                    photo, score = candidate_map[item.photo_id]
                    chosen.append((photo, score, item.reason))
                    if len(chosen) == limit:
                        break
        for photo, score, reason in chosen:
            base["results"].append({
                "photo_id": photo["photo_id"], "caption": photo["caption"],
                "metadata": photo["metadata"], "score": score, "reason": reason,
                "thumbnail_path": photo.get("thumbnail_path", photo["image_path"]),
                "attribution": photo.get("source", {}).get("attribution", ""),
                "source_page_url": photo.get("source", {}).get("page_url", ""),
                "license_url": photo.get("source", {}).get("license_url", ""),
            })
        if not base["results"]:
            base["status"] = "no_match"
        return finish()
