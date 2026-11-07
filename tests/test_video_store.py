"""Video BYTEA/pgvector integration tests; never call an external model API."""

from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vooy_gallery.ai import QueryPlan
from vooy_gallery.store import StoreError
from vooy_gallery.video_ai import SegmentCaption, SegmentRanking
from vooy_gallery.video_search import VideoSearchService, index_videos
from vooy_gallery.video_store import VideoStore


def vector(x=1.0, y=0.0):
    result = np.zeros(1536, dtype=np.float32)
    result[:2] = [x, y]
    return result


def video_input(name="v1", gallery="local-videos", location="서울", at="2026-10-01T00:00:00+09:00"):
    # Container validation belongs to ingestion/ffprobe. Store persists bounded bytes.
    original = b"\x00\x00\x00\x18ftypqt  " + name.encode() + bytes(range(32))
    playback = b"\x00\x00\x00\x18ftypmp42" + name.encode() + bytes(range(48))
    frames = []
    for color in ("blue", "green"):
        buffer = BytesIO()
        Image.new("RGB", (32, 24), color).save(buffer, "JPEG")
        frames.append(buffer.getvalue())
    document = {"video_id": name, "gallery_id": gallery, "title": "개인 산책 영상",
                "sha256": sha256(original).hexdigest(), "duration_ms": 20_000, "mime_type": "video/quicktime",
                "metadata": {"origin": "user_provided", "location": location, "captured_at": at},
                "source": {"attribution": "User video", "identities": ["영상 전체 출연자"]}}
    segments = [{"segment_id": f"{name}-s{i}", "start_ms": i * 10_000, "end_ms": (i + 1) * 10_000,
                 "frame_timestamps_ms": [i * 10_000 + 2500, i * 10_000 + 7500],
                 "frame_sha256s": [sha256(frame).hexdigest() for frame in frames], "frames": frames,
                 "identities": ["가빈"] if i == 0 else []} for i in range(2)]
    return document, original, playback, segments


def put(store, **kwargs):
    document, original, playback, segments = video_input(**kwargs)
    store.put_video(document, original, playback, segments)
    return document, original, playback, segments


def test_postgres_video_bytes_ranges_frames_and_gallery_isolation(pg_store_factory, tmp_path):
    store = VideoStore(pg_store_factory(tmp_path / "cache"))
    document, original, playback, segments = put(store)
    assert store.get_video_asset("v1", "local-videos", "original") == (original, "video/quicktime")
    assert store.get_video_asset("v1", "local-videos", "playback") == (playback, "video/mp4")
    assert store.get_video_asset("v1", "local-videos", "original", 2, 8)[0] == original[2:9]
    assert store.get_video_asset("v1", "local-videos", "playback", 0, 0)[0] == playback[:1]
    assert store.video_asset_info("v1", "local-videos", "playback") == {"size": len(playback), "mime_type": "video/mp4"}
    assert store.get_segment_frames("v1-s0", "local-videos") == segments[0]["frames"]
    assert store.get_segment_poster("v1-s1", "local-videos") == segments[1]["frames"][0]
    assert store.get_video_asset("v1", "local-videos", "poster")[0] == segments[0]["frames"][0]
    merged = store.get_segment("v1-s1", "local-videos")
    assert merged["video_sha256"] == document["sha256"] and merged["metadata"] == document["metadata"]
    assert merged["identities"] == []  # Parent video's names must not become segment identities.
    assert store.get_video("v1", "other") is None and store.get_segment("v1-s0", "other") is None
    assert store.list_videos("other") == store.list_segments("other") == []
    assert store.video_asset_info("v1", "other", "original") is None
    with pytest.raises(ValueError, match="not found"):
        store.get_video_asset("v1", "other", "original")
    with pytest.raises(ValueError, match="not found"):
        store.get_segment_frames("v1-s0", "other")
    with pytest.raises(ValueError, match="not found"):
        store.update_segment("v1-s0", "other", {}, "caption", "embed", vector(), "model")
    with pytest.raises(ValueError, match="range"):
        store.get_video_asset("v1", "local-videos", "playback", -1, 5)


def test_postgres_video_sql_filters_cosine_and_current_cache_only(pg_store_factory, tmp_path):
    store = VideoStore(pg_store_factory(tmp_path / "cache"))
    scenarios = [
        ("good", "서울", "2026-09-30T15:00:00+00:00", vector(.8, .6)),
        ("end", "서울", "2026-11-01T00:00:00+09:00", vector()),
        ("busan", "부산", "2026-10-10T12:00:00+09:00", vector()),
        ("unknown", "서울", None, vector()),
        ("stale", "서울", "2026-10-10T12:00:00+09:00", vector()),
    ]
    eligible = {}
    for name, location, captured, embedding in scenarios:
        put(store, name=name, location=location, at=captured)
        for index in range(2):
            sid = f"{name}-s{index}"
            key = "current-" + sid
            store.update_segment(sid, "local-videos", {"caption_ko": name}, "caption", key, embedding, "model")
            eligible[sid] = "old-key" if name == "stale" else key
    filters = {"location": "서울", "captured_from": "2026-10-01T00:00:00+09:00",
               "captured_until": "2026-11-01T00:00:00+09:00", "people": ["가빈"]}
    rows, count = store.search_segments(vector(), "local-videos", filters, eligible, limit=3)
    assert count == 1 and [row["segment"]["segment_id"] for row in rows] == ["good-s0"]
    assert rows[0]["score"] == pytest.approx(.8, abs=1e-6)
    assert store.count_indexed("local-videos", eligible) == 8
    assert store.count_indexed("other", eligible) == 0
    assert store.search_segments(None, "local-videos", filters, {}, limit=1) == ([], 0)
    metadata_rows, count = store.search_segments(None, "local-videos", filters, eligible, limit=1)
    assert count == 1 and metadata_rows[0]["score"] is None
    all_rows, count = store.search_segments(vector(), "local-videos", {}, eligible, limit=1)
    assert count == 8 and len(all_rows) == 1 and all_rows[0]["score"] == pytest.approx(1)
    with pytest.raises(ValueError, match="timezone"):
        store.search_segments(None, "local-videos", {"captured_from": "2026-10-01"}, eligible)
    with pytest.raises(ValueError, match="1536"):
        store.search_segments([1, 0, 0], "local-videos", {}, eligible)


def test_postgres_video_update_is_atomic_and_reimport_preserves_index(pg_store_factory, tmp_path):
    store = VideoStore(pg_store_factory(tmp_path / "cache"))
    document, original, playback, segments = put(store)
    store.update_segment("v1-s0", "local-videos", {"caption_ko": "기존 설명"}, "c1", "e1", vector(), "model")
    before = store.get_segment("v1-s0", "local-videos")
    # INSERT fails after the caption UPDATE; the transaction must roll back both.
    with pytest.raises(StoreError):
        store.update_segment("v1-s0", "local-videos", {"caption_ko": "실패한 설명"}, "c2", "e2", vector(0, 1), None)
    assert store.get_segment("v1-s0", "local-videos") == before
    assert store.count_indexed("local-videos", {"v1-s0": "e1"}) == 1
    assert store.count_indexed("local-videos", {"v1-s0": "e2"}) == 0
    store.put_video(dict(document, title="재가져오기 제목"), original, playback, segments)
    assert store.get_segment("v1-s0", "local-videos") == before
    assert store.get_video("v1", "local-videos")["title"] == document["title"]
    assert store.get_video_asset("v1", "local-videos", "original")[0] == original
    assert len(store.list_segments("local-videos")) == 2


def test_postgres_video_rejects_invalid_frame_manifest_without_partial_rows(pg_store_factory, tmp_path):
    store = VideoStore(pg_store_factory(tmp_path / "cache"))
    document, original, playback, segments = video_input()
    corrupted = deepcopy(segments)
    corrupted[1]["frame_sha256s"][0] = "0" * 64
    with pytest.raises(ValueError, match="Frame evidence"):
        store.put_video(document, original, playback, corrupted)
    assert store.get_video("v1", "local-videos") is None and store.list_segments("local-videos") == []
    overlapping = deepcopy(segments)
    overlapping[1]["start_ms"] = 9000
    with pytest.raises(ValueError, match="overlapping"):
        store.put_video(document, original, playback, overlapping)
    assert store.get_video("v1", "local-videos") is None


def test_postgres_video_index_resume_and_search_end_to_end_without_api(pg_store_factory, tmp_path):
    store = VideoStore(pg_store_factory(tmp_path / "cache"))
    put(store)
    config = SimpleNamespace(gallery_id="local-videos", vision_model="vision-test", embedding_model="embed-test",
                             embedding_dimensions=1536)

    class AI:
        captions = 0
        embeddings = 0

        def caption_segment(self, segment, frames):
            self.captions += 1
            assert len(frames) == 2
            return SegmentCaption(caption_ko="푸른 숲에서 흰 옷을 입고 서 있음", scene="숲", observed_actions=["서 있음"],
                                  clothing=["흰 옷"], background=["나무"], tags=["숲"], people_count=1,
                                  limitations=["샘플 사이 동작은 미상"])

        def embed(self, text):
            self.embeddings += 1
            return vector()

        def plan_query(self, query, now, locations, known_people=None):
            assert known_people == ["가빈"]
            return QueryPlan(status="ok", semantic_query="흰 옷", location="서울", captured_from=None,
                             captured_until=None, people=["가빈"], clarification=None)

        def rerank_segments(self, query, semantic_query, segments):
            assert [segment["segment_id"] for segment in segments] == ["v1-s0"]
            return SegmentRanking(segments=[{"segment_id": "v1-s0", "relevant": True, "reason": "흰 옷 관찰"}])

    ai = AI()
    assert index_videos(store, config, ai=ai)["indexed"] == 2
    assert index_videos(store, config, ai=ai)["reused"] == 2
    assert ai.captions == ai.embeddings == 2
    result = VideoSearchService(store, config, ai).search("서울 가빈 흰 옷 영상")
    assert result["status"] == "ok" and result["filtered_count"] == 1
    assert result["results"][0]["segment_id"] == "v1-s0"
    assert result["results"][0]["frame_timestamps_ms"] == [2500, 7500]
    assert store.base.get_search(result["result_id"])["results"] == result["results"]
    assert store.base.usage_summary()["groups"] == []
