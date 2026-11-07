"""Resumable segment indexing and SQL-filtered video retrieval."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime
from typing import Any

from openai import APIError, AuthenticationError, BadRequestError, NotFoundError, PermissionDeniedError

from .ai import ConfigurationError, ModelResponseError, QueryPlan, normalized_vector
from .retrieval import KST, canonical_location, parse_time
from .video_ai import (
    SegmentCaption,
    SegmentRanking,
    VideoAI,
    segment_caption_key,
    segment_embedding_key,
    segment_embedding_text,
)

FATAL_ERRORS = (ConfigurationError, AuthenticationError, PermissionDeniedError, NotFoundError, BadRequestError)


def index_videos(video_store: Any, settings: Any, *, limit: int | None = None, ai: Any = None) -> dict:
    """Index all segments of the first ``limit`` videos, preserving retryable work."""
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive video count")
    videos = video_store.list_videos(settings.gallery_id)
    if limit is not None:
        videos = videos[:limit]
    if not videos:
        raise ConfigurationError("동영상이 없습니다. 먼저 동영상을 가져오세요.")
    selected = {video["video_id"] for video in videos}
    segments = [s for s in video_store.list_segments(settings.gallery_id) if s["video_id"] in selected]
    if not segments:
        raise ConfigurationError("추출된 동영상 구간이 없습니다. 프레임 추출 상태를 확인하세요.")
    service = ai if ai is not None else VideoAI(video_store.base, settings)
    stats = {"status": "ok", "gallery_id": settings.gallery_id, "requested_videos": len(videos),
             "requested_segments": len(segments), "indexed": 0, "reused": 0, "excluded": 0, "failed": [],
             "usage_before": video_store.base.usage_summary()}
    for segment in segments:
        if segment.get("review", {}).get("included") is False:
            stats["excluded"] += 1
            continue
        try:
            caption_key = segment_caption_key(segment, settings)
            if segment.get("caption_cache_key") == caption_key and segment.get("caption"):
                caption = SegmentCaption.model_validate(segment["caption"]).model_dump()
            else:
                cached = video_store.base.get_cache(caption_key)
                if cached is None:
                    frames = video_store.get_segment_frames(segment["segment_id"], settings.gallery_id)
                    if len(frames) != 2 or [hashlib.sha256(frame).hexdigest() for frame in frames] != segment["frame_sha256s"]:
                        raise ModelResponseError("Stored frame bytes do not match the segment manifest.")
                    cached = service.caption_segment(segment, frames)
                caption = SegmentCaption.model_validate(cached).model_dump()
                # Save the expensive caption before embedding, even if the latter fails.
                video_store.base.set_cache(caption_key, caption)
            document = dict(segment, caption=caption)
            embedding_key = segment_embedding_key(document, settings)
            if (segment.get("index_status") == "indexed"
                    and segment.get("caption_cache_key") == caption_key
                    and segment.get("embedding_cache_key") == embedding_key
                    and video_store.count_indexed(settings.gallery_id, {segment["segment_id"]: embedding_key})):
                stats["indexed"] += 1
                stats["reused"] += 1
                continue
            vector = normalized_vector(service.embed(segment_embedding_text(document)), settings.embedding_dimensions)
            video_store.update_segment(segment["segment_id"], settings.gallery_id, caption, caption_key,
                                       embedding_key, vector, settings.embedding_model)
            stats["indexed"] += 1
        except FATAL_ERRORS:
            raise
        except (APIError, OSError, RuntimeError, ValueError) as exc:
            stats["status"] = "partial"
            stats["failed"].append({"segment_id": segment["segment_id"], "error_type": type(exc).__name__,
                                    "error": "구간 처리에 실패했습니다. 저장된 작업을 재사용해 다시 시도할 수 있습니다."})
    stats["usage_after"] = video_store.base.usage_summary()
    return stats


class VideoSearchService:
    def __init__(self, video_store: Any, settings: Any, ai: Any = None):
        self.store = video_store
        self.settings = settings
        self.ai = ai if ai is not None else VideoAI(video_store.base, settings)

    def search(self, query: str, gallery_id: str | None = None, limit: int = 3,
               now: datetime | None = None) -> dict:
        started = time.monotonic()
        gallery_id = gallery_id or self.settings.gallery_id
        if gallery_id != self.settings.gallery_id:
            raise ValueError("This local service only serves its configured gallery")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 3:
            raise ValueError("limit must be between 1 and 3")
        if not isinstance(query, str) or not query.strip() or len(query) > 4000:
            raise ValueError("검색어는 1~4,000자여야 합니다.")
        query = query.strip()
        now = now or datetime.now(KST)
        if now.tzinfo is None:
            raise ValueError("now must include its timezone")
        now = now.astimezone(KST)
        segments = self.store.list_segments(gallery_id)
        if not segments:
            raise ConfigurationError("검색할 동영상 구간이 없습니다. 가져오기와 인덱싱을 먼저 실행하세요.")
        eligible = {}
        for segment in segments:
            if (segment.get("index_status") != "indexed" or not segment.get("caption")
                    or segment.get("review", {}).get("included") is False):
                continue
            if segment.get("caption_cache_key") == segment_caption_key(segment, self.settings):
                eligible[segment["segment_id"]] = segment_embedding_key(segment, self.settings)
        indexed_count = self.store.count_indexed(gallery_id, eligible)
        if not indexed_count:
            raise ConfigurationError("현재 모델과 프레임 설정으로 인덱싱된 동영상 구간이 없습니다.")
        locations = sorted({s["metadata"]["location"] for s in segments
                            if s["segment_id"] in eligible and (s.get("metadata") or {}).get("location")})
        known_people = sorted({name for segment in segments if segment["segment_id"] in eligible
                               for name in segment.get("identities", [])})
        plan = QueryPlan.model_validate(self.ai.plan_query(query, now, locations, known_people=known_people))
        result = {"status": "ok", "media_type": "video", "query": query, "gallery_id": gallery_id,
                  "results": [], "clarification": None, "requested_limit": limit, "now": now.isoformat(),
                  "indexed_count": indexed_count, "semantic_query": plan.semantic_query,
                  "evidence_scope": "sampled_frames",
                  "evidence_note": "구간 내 샘플 프레임의 관찰 결과이며, 행동이 구간 전체에서 지속됨을 뜻하지 않습니다.",
                  "applied_filters": {"location": canonical_location(plan.location) if plan.location else None,
                                      "captured_from": plan.captured_from, "captured_until": plan.captured_until,
                                      "people": list(dict.fromkeys(name.strip() for name in plan.people if name.strip()))}}

        def finish() -> dict:
            result["elapsed_seconds"] = round(time.monotonic() - started, 4)
            return self.store.base.save_search(result)

        if plan.status == "needs_clarification":
            result["status"] = "needs_clarification"
            result["clarification"] = plan.clarification or "어떤 영상의 어느 상황을 찾으시나요?"
            return finish()
        try:
            start, end = parse_time(plan.captured_from), parse_time(plan.captured_until)
            if start is not None and end is not None and start >= end:
                raise ValueError("invalid date range")
        except (ValueError, TypeError):
            result["status"] = "needs_clarification"
            result["clarification"] = "찾을 영상의 촬영 날짜 범위를 다시 알려주세요."
            return finish()
        rows, count = self.store.search_segments(None, gallery_id, result["applied_filters"], eligible,
                                                limit=limit if not plan.semantic_query.strip() else 1)
        result["filtered_count"] = count
        if not count:
            result["status"] = "no_match"
            return finish()
        chosen = []
        if not plan.semantic_query.strip():
            chosen = [(row["segment"], None, "요청한 촬영 메타데이터 조건과 일치합니다.") for row in rows[:limit]]
        else:
            vector = normalized_vector(self.ai.embed(plan.semantic_query), self.settings.embedding_dimensions)
            rows, _ = self.store.search_segments(vector.tolist(), gallery_id, result["applied_filters"], eligible, limit=10)
            candidates = {row["segment"]["segment_id"]: row for row in rows}
            ranking = SegmentRanking.model_validate(self.ai.rerank_segments(
                query, plan.semantic_query, [row["segment"] for row in rows]))
            seen = set()
            for item in ranking.segments:
                if item.segment_id in seen or item.segment_id not in candidates:
                    continue
                seen.add(item.segment_id)
                if item.relevant:
                    row = candidates[item.segment_id]
                    chosen.append((row["segment"], row["score"], item.reason))
                    if len(chosen) == limit:
                        break
        for segment, score, reason in chosen:
            source = segment.get("source") or {}
            result["results"].append({
                "segment_id": segment["segment_id"], "video_id": segment["video_id"],
                "start_ms": segment["start_ms"], "end_ms": segment["end_ms"],
                "frame_timestamps_ms": segment["frame_timestamps_ms"], "caption": segment["caption"],
                "evidence_scope": "sampled_frames",
                "metadata": segment.get("metadata") or {}, "score": score, "reason": reason,
                "video_title": segment.get("video_title", ""), "source": source,
                "attribution": source.get("attribution", ""), "source_page_url": source.get("page_url", ""),
            })
        if not result["results"]:
            result["status"] = "no_match"
        return finish()
