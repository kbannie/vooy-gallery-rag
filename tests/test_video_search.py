from copy import deepcopy
from datetime import datetime
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vooy_gallery.ai import ConfigurationError, ModelResponseError, QueryPlan
from vooy_gallery.retrieval import matches_filters
from vooy_gallery.video_ai import (
    SegmentCaption,
    SegmentRanking,
    VideoAI,
    segment_caption_key,
    segment_embedding_key,
    segment_embedding_text,
)
from vooy_gallery.video_search import VideoSearchService, index_videos

CAPTION = {"caption_ko": "흰 상의를 입은 사람이 나무 옆에 서 있다.", "scene": "공원",
           "observed_actions": ["서 있음"], "clothing": ["흰 상의"], "background": ["나무"],
           "tags": ["공원", "나무"], "people_count": 1,
           "limitations": ["두 샘플 사이의 이동 여부와 소리는 확인할 수 없음"]}


def settings():
    return SimpleNamespace(gallery_id="videos", vision_model="vision-test", embedding_model="embed-test",
                           embedding_dimensions=3, reasoning_effort="none", api_key=None, api_timeout=10,
                           vision_input_usd_per_million=.10, vision_output_usd_per_million=.50,
                           embedding_input_usd_per_million=.02)


def plan(**kwargs):
    return QueryPlan(**dict({"status": "ok", "semantic_query": "", "location": None,
                            "captured_from": None, "captured_until": None, "people": [],
                            "clarification": None}, **kwargs))


class BaseStore:
    def __init__(self):
        self.cache = {}
        self.usages = []
        self.searches = []

    def get_cache(self, key):
        return deepcopy(self.cache.get(key))

    def set_cache(self, key, value):
        self.cache[key] = deepcopy(value)

    def usage_summary(self):
        return {"calls": len(self.usages), "total_estimated_usd": 0}

    def record_usage(self, *args):
        self.usages.append(args)

    def save_search(self, result):
        result = dict(result, result_id=str(len(self.searches) + 1))
        self.searches.append(deepcopy(result))
        return result


class VideoStore:
    """SQL test double; production filtering and vector ranking belong to PostgreSQL."""

    def __init__(self):
        self.base = BaseStore()
        self.segments = {}
        self.frames = {}
        self.vectors = {}
        self.queries = []
        self.frame_reads = 0

    def list_videos(self, gallery):
        return [{"video_id": name} for name in sorted({s["video_id"] for s in self.list_segments(gallery)})]

    def list_segments(self, gallery):
        return [deepcopy(s) for s in self.segments.values() if s["gallery_id"] == gallery]

    def get_segment_frames(self, segment_id, gallery):
        assert self.segments[segment_id]["gallery_id"] == gallery
        self.frame_reads += 1
        return self.frames[segment_id]

    def update_segment(self, segment_id, gallery, caption, caption_key, embedding_key, vector, model):
        assert self.segments[segment_id]["gallery_id"] == gallery
        self.segments[segment_id].update(caption=deepcopy(caption), caption_cache_key=caption_key,
                                         embedding_cache_key=embedding_key, embedding_model=model,
                                         index_status="indexed")
        self.vectors[segment_id] = (embedding_key, np.asarray(vector))

    def count_indexed(self, gallery, eligible):
        return sum(self.segments[key]["gallery_id"] == gallery and key in self.vectors
                   and self.vectors[key][0] == expected for key, expected in eligible.items())

    def search_segments(self, vector, gallery, filters, eligible, limit=10):
        self.queries.append((vector, deepcopy(filters), limit))
        candidates = [s for s in self.list_segments(gallery) if s["segment_id"] in eligible
                      and s["segment_id"] in self.vectors
                      and self.vectors[s["segment_id"]][0] == eligible[s["segment_id"]]
                      and set(filters.get("people", [])).issubset(s["identities"])
                      and matches_filters(s, plan(**dict(filters, people=[])))]
        if vector is None:
            candidates.sort(key=lambda s: s["metadata"].get("captured_at") or "", reverse=True)
            rows = [{"segment": s, "score": None} for s in candidates]
        else:
            rows = [{"segment": s, "score": float(np.dot(self.vectors[s["segment_id"]][1], vector))}
                    for s in candidates]
            rows.sort(key=lambda row: row["score"], reverse=True)
        return rows[:limit], len(candidates)


def add_segment(store, config, name="one", video="video-one", start=0, location="서울", indexed=True,
                at="2026-10-10T12:00:00+09:00", identities=None):
    frames = []
    for color in ("red", "blue"):
        data = BytesIO()
        Image.new("RGB", (16, 16), color).save(data, "JPEG")
        frames.append(data.getvalue())
    segment = {"segment_id": name, "video_id": video, "gallery_id": config.gallery_id,
               "video_sha256": sha256(video.encode()).hexdigest(), "video_title": "1999 Paris BTS source title",
               "start_ms": start, "end_ms": start + 10_000,
               "frame_timestamps_ms": [start + 2500, start + 7500],
               "frame_sha256s": [sha256(frame).hexdigest() for frame in frames],
               "identities": identities or [], "source": {"attribution": "Source author", "identities": ["RM"]},
               "metadata": {"origin": "synthetic", "location": location, "captured_at": at}}
    store.segments[name] = segment
    store.frames[name] = frames
    if indexed:
        document = dict(segment, caption=CAPTION)
        store.update_segment(name, config.gallery_id, CAPTION, segment_caption_key(document, config),
                             segment_embedding_key(document, config), [1, 0, 0], config.embedding_model)
    return segment


class FakeAI:
    def __init__(self, query_plan=None, ranking=None):
        self.plan = query_plan or plan()
        self.ranking = ranking
        self.caption_calls = self.embed_calls = 0
        self.rerank_ids = []
        self.now = None
        self.known_people = None

    def caption_segment(self, segment, frames):
        self.caption_calls += 1
        return SegmentCaption(**CAPTION)

    def embed(self, text):
        self.embed_calls += 1
        return [1, 0, 0]

    def plan_query(self, query, now, locations, known_people=None):
        self.now = now
        self.known_people = known_people
        return self.plan

    def rerank_segments(self, query, semantic_query, segments):
        self.rerank_ids = [s["segment_id"] for s in segments]
        return self.ranking or SegmentRanking(segments=[
            {"segment_id": s["segment_id"], "relevant": True, "reason": "나무 옆에 서 있음"} for s in segments])


def test_video_embedding_is_visual_and_never_copies_video_level_cast_or_metadata():
    config, store = settings(), VideoStore()
    segment = add_segment(store, config)
    text = segment_embedding_text(segment)
    assert "흰 상의" in text
    assert all(term not in text for term in ("BTS", "RM", "1999", "Paris", "서울", "2026"))
    assert not segment["identities"]


def test_sdk_video_caption_uses_two_timestamped_frames_and_caches():
    config, store = settings(), VideoStore()
    segment = add_segment(store, config, indexed=False)
    requests = []

    def parse(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(output_parsed=SegmentCaption(**CAPTION), status="completed",
                               usage={"input_tokens": 200, "output_tokens": 50})

    ai = VideoAI(store.base, config, client=SimpleNamespace(responses=SimpleNamespace(parse=parse)))
    first = ai.caption_segment(segment, store.frames["one"])
    assert first == ai.caption_segment(segment, store.frames["one"])
    assert len(requests) == len(store.base.usages) == 1
    content = requests[0]["input"][1]["content"]
    assert sum(item["type"] == "input_image" for item in content) == 2
    assert "2500ms" in content[1]["text"] and "7500ms" in content[3]["text"]
    assert "BTS" not in str(content) and "Paris" not in str(content)
    assert "연속 동작" in requests[0]["input"][0]["content"]


def test_video_planner_has_generic_private_annotations_and_separate_cache():
    config, store = settings(), VideoStore()
    requests = []

    def parse(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(output_parsed=plan(people=["지민"]), status="completed",
                               usage={"input_tokens": 100, "output_tokens": 20})

    ai = VideoAI(store.base, config, client=SimpleNamespace(responses=SimpleNamespace(parse=parse)))
    now = datetime.fromisoformat("2026-11-07T18:00:00+09:00")
    first = ai.plan_query("지민 영상", now, ["서울"], known_people=["지민"])
    assert first.people == ["지민"]
    assert ai.plan_query("지민 영상", now, ["서울"], known_people=["지민"]) == first
    assert len(requests) == 1 and store.base.usages[0][0] == "video_query_plan"
    request = requests[0]["input"]
    assert '"annotated_people": ["지민"]' in request[1]["content"]
    assert "BTS" not in str(request) and "Jimin" not in str(request) and "Jung Kook" not in str(request)
    # A new annotation set must not reuse an earlier query-plan cache.
    ai.plan_query("지민 영상", now, ["서울"], known_people=[])
    assert len(requests) == 2


def test_frame_corruption_does_not_call_provider():
    config, store = settings(), VideoStore()
    segment = add_segment(store, config, indexed=False)
    ai = FakeAI()
    store.frames["one"][0] = b"corrupt"
    report = index_videos(store, config, ai=ai)
    assert report["status"] == "partial" and len(report["failed"]) == 1
    assert ai.caption_calls == ai.embed_calls == 0
    with pytest.raises(ModelResponseError, match="manifest"):
        VideoAI(store.base, config).caption_segment(segment, store.frames["one"])


def test_index_resume_reuses_caption_after_embedding_error_and_skips_completed():
    config, store = settings(), VideoStore()
    add_segment(store, config, indexed=False)
    ai = FakeAI()
    embed = ai.embed
    ai.embed = lambda text: (_ for _ in ()).throw(RuntimeError("provider unavailable"))
    failed = index_videos(store, config, ai=ai)
    assert failed["status"] == "partial" and ai.caption_calls == 1
    assert "provider unavailable" not in str(failed["failed"])
    ai.embed = embed
    assert index_videos(store, config, ai=ai)["indexed"] == 1
    assert index_videos(store, config, ai=ai)["reused"] == 1
    assert ai.caption_calls == ai.embed_calls == store.frame_reads == 1


def test_index_limit_counts_videos_and_keeps_segment_identity_empty():
    config, store = settings(), VideoStore()
    add_segment(store, config, "a0", "a", indexed=False)
    add_segment(store, config, "a1", "a", start=10_000, indexed=False)
    add_segment(store, config, "b0", "b", indexed=False)
    report = index_videos(store, config, limit=1, ai=FakeAI())
    assert report["requested_videos"] == 1 and report["requested_segments"] == report["indexed"] == 2
    assert "index_status" not in store.segments["b0"]
    assert all(not s["identities"] for s in store.segments.values())


def test_video_search_sql_filters_before_reranking_and_returns_exact_timestamps():
    config, store = settings(), VideoStore()
    add_segment(store, config, "match", at="2026-09-30T15:00:00+00:00", identities=["지민"])
    add_segment(store, config, "end", at="2026-11-01T00:00:00+09:00", identities=["지민"])
    add_segment(store, config, "elsewhere", location="부산", identities=["지민"])
    add_segment(store, config, "unknown-person")
    ai = FakeAI(plan(semantic_query="흰 상의", location="서울시", people=["지민"],
                     captured_from="2026-10-01T00:00:00+09:00", captured_until="2026-11-01T00:00:00+09:00"))
    result = VideoSearchService(store, config, ai).search("지난달 서울 지민 흰 상의 영상", now=datetime.fromisoformat(
        "2026-11-07T18:00:00+09:00"))
    assert ai.rerank_ids == ["match"]
    assert [row["segment_id"] for row in result["results"]] == ["match"]
    assert result["results"][0]["start_ms"] == 0 and result["results"][0]["end_ms"] == 10_000
    assert store.queries[0][0] is None and store.queries[1][2] == 10
    assert ai.now.isoformat() == "2026-11-07T18:00:00+09:00"
    assert result["result_id"] == "1" and result["media_type"] == "video"
    assert result["applied_filters"]["people"] == ["지민"] and ai.known_people == ["지민"]
    assert result["evidence_scope"] == result["results"][0]["evidence_scope"] == "sampled_frames"


def test_private_annotation_is_not_replaced_by_celebrity_alias():
    config, store = settings(), VideoStore()
    add_segment(store, config, "literal-jimin", identities=["Jimin"])
    ai = FakeAI(plan(people=["지민"]))
    result = VideoSearchService(store, config, ai).search("지민 영상")
    assert result["status"] == "no_match"
    assert result["applied_filters"]["people"] == ["지민"]
    assert ai.known_people == ["Jimin"] and ai.embed_calls == 0


def test_grounded_reranker_rejects_unknown_and_duplicate_segment_ids_without_padding():
    config, store = settings(), VideoStore()
    add_segment(store, config)
    ranking = SegmentRanking(segments=[
        {"segment_id": "not-in-candidates", "relevant": True, "reason": "invented"},
        {"segment_id": "one", "relevant": True, "reason": "visible"},
        {"segment_id": "one", "relevant": True, "reason": "duplicate"},
    ])
    result = VideoSearchService(store, config, FakeAI(plan(semantic_query="나무"), ranking)).search("나무 영상")
    assert len(result["results"]) == 1 and result["results"][0]["segment_id"] == "one"


def test_no_match_for_unobserved_action_and_for_empty_metadata_filter():
    config, store = settings(), VideoStore()
    add_segment(store, config)
    ranking = SegmentRanking(segments=[{"segment_id": "one", "relevant": False, "reason": "뛰는 동작 근거 없음"}])
    ai = FakeAI(plan(semantic_query="달리기"), ranking)
    assert VideoSearchService(store, config, ai).search("달리는 영상")["status"] == "no_match"
    ai = FakeAI(plan(semantic_query="달리기", location="부산"))
    assert VideoSearchService(store, config, ai).search("부산 달리기")["status"] == "no_match"
    assert ai.embed_calls == 0


@pytest.mark.parametrize("change", ["vision_model", "embedding_model"])
def test_video_model_drift_is_configuration_error_not_no_match(change):
    config, store = settings(), VideoStore()
    add_segment(store, config)
    setattr(config, change, "new-model")
    with pytest.raises(ConfigurationError, match="현재 모델"):
        VideoSearchService(store, config, FakeAI()).search("영상")


def test_video_clarification_and_metadata_only_search_need_no_query_embedding():
    config, store = settings(), VideoStore()
    add_segment(store, config)
    ai = FakeAI(plan(status="needs_clarification", clarification="어느 장소인가요?"))
    assert VideoSearchService(store, config, ai).search("거기 영상")["status"] == "needs_clarification"
    assert ai.embed_calls == 0 and not store.queries
    ai = FakeAI(plan(location="서울"))
    result = VideoSearchService(store, config, ai).search("서울 영상")
    assert result["status"] == "ok" and result["results"][0]["score"] is None
    assert ai.embed_calls == 0 and not ai.rerank_ids


def test_segment_cache_changes_with_sampling_manifest_and_boundaries():
    config, store = settings(), VideoStore()
    original = add_segment(store, config)
    original_key = segment_caption_key(original, config)
    changed = deepcopy(original)
    changed["frame_timestamps_ms"][0] += 1
    assert segment_caption_key(changed, config) != original_key
    changed = deepcopy(original)
    changed["frame_sha256s"][0] = "f" * 64
    assert segment_caption_key(changed, config) != original_key
    changed = deepcopy(original)
    changed["end_ms"] -= 1
    assert segment_caption_key(changed, config) != original_key
