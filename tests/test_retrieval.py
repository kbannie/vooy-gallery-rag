from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from threading import Barrier, Lock, get_ident
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vooy_gallery.ai import (
    Caption,
    ConfigurationError,
    GalleryAI,
    ModelResponseError,
    QueryPlan,
    Ranking,
    caption_key,
    embedding_key,
    embedding_text,
)
from vooy_gallery.retrieval import SearchService, index_gallery, matches_filters


def settings(tmp_path):
    return SimpleNamespace(
        data_dir=tmp_path, gallery_id="bts-demo", api_key=None,
        vision_model="gpt-6-luna", embedding_model="text-embedding-3-small",
        embedding_dimensions=3, api_timeout=10, reasoning_effort="none",
        vision_input_usd_per_million=0.10, vision_output_usd_per_million=0.50,
        embedding_input_usd_per_million=0.02,
    )


CAPTION = {
    "caption_ko": "검은 재킷을 입은 사람이 마이크를 들고 노래한다.",
    "scene": "무대", "clothing": ["검은 재킷"], "actions": ["노래"],
    "background": ["무대 조명"], "tags": ["마이크", "공연"],
    "people_count": 1, "is_photo": True, "contains_people": True,
}


def plan(**kwargs):
    values = {"status": "ok", "semantic_query": "", "location": None, "captured_from": None,
              "captured_until": None, "people": [], "clarification": None}
    values.update(kwargs)
    return QueryPlan(**values)


class MemoryStore:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.photos = {}
        self.cache = {}
        self.vectors = {}
        self.usages = []
        self.searches = []
        self.candidate_queries = []
        self.embedding_reads = 0

    def upsert_photo(self, photo):
        self.photos[photo["photo_id"]] = deepcopy(photo)

    def list_photos(self, gallery_id):
        return [deepcopy(self.photos[key]) for key in sorted(self.photos)
                if self.photos[key]["gallery_id"] == gallery_id]

    def get_cache(self, key):
        return self.cache.get(key)

    def set_cache(self, key, value):
        self.cache[key] = deepcopy(value)

    def get_embedding(self, photo_id, cache_key=None):
        self.embedding_reads += 1
        record = self.vectors.get(photo_id)
        if record and (cache_key is None or record[0] == cache_key):
            return record[1]
        return None

    def get_asset(self, photo_id, gallery_id="bts-demo", thumbnail=False):
        photo = self.photos[photo_id]
        assert photo["gallery_id"] == gallery_id
        return (self.data_dir / photo["image_path"]).read_bytes(), "image/jpeg"

    def count_indexed(self, gallery_id, eligible_cache_keys):
        return sum(
            self.photos[photo_id]["gallery_id"] == gallery_id and photo_id in self.vectors
            and self.vectors[photo_id][0] == key
            for photo_id, key in eligible_cache_keys.items()
        )

    def search_candidates(self, query_vector, gallery_id, filters, eligible_cache_keys, limit=10):
        """Test double for SQL. Production ranking is entirely in PostgreSQL/pgvector."""
        self.candidate_queries.append((query_vector, deepcopy(filters), limit))
        query_plan = plan(**filters)
        eligible = [
            deepcopy(photo) for photo in self.photos.values()
            if photo["gallery_id"] == gallery_id
            and photo["photo_id"] in eligible_cache_keys
            and photo["photo_id"] in self.vectors
            and self.vectors[photo["photo_id"]][0] == eligible_cache_keys[photo["photo_id"]]
            and matches_filters(photo, query_plan)
        ]
        if query_vector is None:
            eligible.sort(key=lambda photo: photo["photo_id"])
            eligible.sort(key=lambda photo: photo["metadata"].get("captured_at") or "", reverse=True)
            return [{"photo": photo, "score": None} for photo in eligible[:limit]], len(eligible)
        scored = [{"photo": photo, "score": float(np.dot(self.vectors[photo["photo_id"]][1], query_vector))}
                  for photo in eligible]
        scored.sort(key=lambda item: (-item["score"], item["photo"]["photo_id"]))
        return scored[:limit], len(eligible)

    def set_embedding(self, photo_id, vector, cache_key, model):
        self.vectors[photo_id] = (cache_key, np.asarray(vector))

    def record_usage(self, operation, model, usage, input_price, output_price=0):
        self.usages.append((operation, model, usage, input_price, output_price))

    def usage_summary(self):
        return {"calls": len(self.usages)}

    def save_search(self, result):
        result = deepcopy(result)
        result["result_id"] = str(len(self.searches) + 1)
        self.searches.append(result)
        return result


def add_photo(store, config, photo_id="one", location="서울", at="2026-10-10T12:00:00+09:00",
              identities=None, indexed=True, vector=None):
    photo = {
        "photo_id": photo_id, "gallery_id": "bts-demo", "sha256": photo_id,
        "image_path": photo_id + ".jpg", "thumbnail_path": photo_id + "-thumb.jpg",
        "identities": identities or ["Jimin"],
        "source": {"title": "2018 New York concert", "description": "Times Square 2018",
                   "attribution": "Photographer · CC BY 4.0", "page_url": "https://example.test/photo"},
        "metadata": {"origin": "synthetic", "location": location, "captured_at": at,
                     "event": "가상 이벤트", "latitude": None, "longitude": None},
    }
    if indexed:
        photo["caption"] = deepcopy(CAPTION)
        photo["caption_cache_key"] = caption_key(photo, config)
        photo["index_status"] = "indexed"
        key = embedding_key(photo, config)
        store.set_embedding(photo_id, [1, 0, 0] if vector is None else vector, key, config.embedding_model)
        photo["embedding_cache_key"] = key
    store.upsert_photo(photo)
    return photo


class FakeAI:
    def __init__(self, query_plan=None, ranking=None):
        self.query_plan = query_plan or plan()
        self.ranking = ranking
        self.caption_calls = 0
        self.embed_calls = 0
        self.rerank_inputs = []
        self.seen_now = None

    def caption(self, photo):
        self.caption_calls += 1
        return Caption(**CAPTION)

    def embed(self, text):
        self.embed_calls += 1
        return np.array([1, 0, 0], dtype=np.float32)

    def plan_query(self, query, now, locations):
        self.seen_now = now
        return self.query_plan

    def rerank(self, query, semantic_query, candidates):
        self.rerank_inputs.append(candidates)
        return self.ranking or Ranking(photos=[
            {"photo_id": item["photo_id"], "relevant": True, "reason": "검은 재킷과 마이크가 보임"}
            for item in candidates
        ])


def test_embedding_has_visual_and_identity_evidence_but_no_original_or_mock_metadata(tmp_path):
    config = settings(tmp_path)
    photo = add_photo(MemoryStore(tmp_path), config)
    text = embedding_text(photo)
    assert "검은 재킷" in text and "Jimin" in text
    for forbidden in ("2018", "New York", "Times Square", "서울", "2026", "가상 이벤트"):
        assert forbidden not in text


def test_manual_daily_photo_exclusion_survives_reindex_and_search(tmp_path):
    config = settings(tmp_path)
    store = MemoryStore(tmp_path)
    rejected = add_photo(store, config, "stage-photo")
    rejected["review"] = {"included": False, "reason": "공연 사진은 일상 갤러리에서 제외"}
    store.upsert_photo(rejected)
    add_photo(store, config, "daily-photo")
    fake = FakeAI()
    stats = index_gallery(store, config, client=fake)
    assert stats["excluded"] == 1 and stats["indexed"] == 1
    assert fake.caption_calls == 0 and fake.embed_calls == 0
    assert store.photos["stage-photo"]["index_status"] == "excluded"
    result = SearchService(store, config, client=fake).search("사진")
    assert [row["photo_id"] for row in result["results"]] == ["daily-photo"]


def test_time_filter_is_inclusive_start_exclusive_end_and_handles_offset(tmp_path):
    config = settings(tmp_path)
    store = MemoryStore(tmp_path)
    photo = add_photo(store, config, at="2026-09-30T15:00:00+00:00")
    october = plan(captured_from="2026-10-01T00:00:00+09:00", captured_until="2026-11-01T00:00:00+09:00")
    assert matches_filters(photo, october)
    photo["metadata"]["captured_at"] = "2026-10-31T15:00:00+00:00"
    assert not matches_filters(photo, october)
    photo["metadata"]["captured_at"] = "2026-10-05T10:00:00"  # unknown EXIF timezone
    assert not matches_filters(photo, october)
    photo["metadata"]["captured_at"] = None
    assert not matches_filters(photo, october)


def test_metadata_filters_precede_retrieval_and_cannot_leak_other_city(tmp_path):
    config = settings(tmp_path)
    store = MemoryStore(tmp_path)
    add_photo(store, config, "seoul", "서울", vector=[0.8, 0.6, 0])
    add_photo(store, config, "busan", "부산", vector=[1, 0, 0])
    fake = FakeAI(plan(semantic_query="검은 재킷", location="서울시", people=["지민"]))
    result = SearchService(store, config, client=fake).search("서울에서 지민이 검은 재킷 입은 사진")
    assert [item["photo_id"] for item in result["results"]] == ["seoul"]
    assert [item["photo_id"] for item in fake.rerank_inputs[0]] == ["seoul"]
    assert result["applied_filters"]["location"] == "서울"
    assert result["applied_filters"]["people"] == ["Jimin"]


def test_metadata_only_search_sorts_recent_without_embedding_or_reranking(tmp_path):
    config = settings(tmp_path)
    store = MemoryStore(tmp_path)
    add_photo(store, config, "older", at="2026-09-10T12:00:00+09:00")
    add_photo(store, config, "newer", at="2026-10-10T12:00:00+09:00")
    fake = FakeAI(plan(location="서울"))
    result = SearchService(store, config, client=fake).search("서울 사진", limit=1)
    assert result["results"][0]["photo_id"] == "newer"
    assert fake.embed_calls == 0 and not fake.rerank_inputs


def test_hallucinated_ids_and_duplicates_are_never_returned(tmp_path):
    config = settings(tmp_path)
    store = MemoryStore(tmp_path)
    add_photo(store, config)
    ranking = Ranking(photos=[
        {"photo_id": "not-in-candidates", "relevant": True, "reason": "invented"},
        {"photo_id": "one", "relevant": True, "reason": "valid"},
        {"photo_id": "one", "relevant": True, "reason": "duplicate"},
    ])
    result = SearchService(store, config, client=FakeAI(plan(semantic_query="노래"), ranking)).search("노래 사진")
    assert result["status"] == "ok"
    assert [item["photo_id"] for item in result["results"]] == ["one"]


def test_no_match_has_no_padding(tmp_path):
    config = settings(tmp_path)
    store = MemoryStore(tmp_path)
    add_photo(store, config)
    ranking = Ranking(photos=[{"photo_id": "one", "relevant": False, "reason": "강아지 없음"}])
    result = SearchService(store, config, client=FakeAI(plan(semantic_query="강아지"), ranking)).search("강아지 사진")
    assert result["status"] == "no_match" and result["results"] == []


def test_empty_metadata_filter_stops_before_embedding(tmp_path):
    config = settings(tmp_path)
    store = MemoryStore(tmp_path)
    add_photo(store, config)
    fake = FakeAI(plan(semantic_query="노래", location="파리"))
    result = SearchService(store, config, client=fake).search("파리 노래")
    assert result["status"] == "no_match" and not fake.embed_calls


def test_ambiguity_is_separate_from_no_match(tmp_path):
    config = settings(tmp_path)
    store = MemoryStore(tmp_path)
    add_photo(store, config)
    fake = FakeAI(plan(status="needs_clarification", clarification="어느 장소인가요?"))
    result = SearchService(store, config, client=fake).search("거기 사진")
    assert result["status"] == "needs_clarification" and result["clarification"] == "어느 장소인가요?"
    assert not fake.embed_calls


def test_relative_date_anchor_reaches_planner_in_kst(tmp_path):
    config = settings(tmp_path)
    store = MemoryStore(tmp_path)
    add_photo(store, config)
    fake = FakeAI(plan(captured_from="2026-10-01T00:00:00+09:00", captured_until="2026-11-01T00:00:00+09:00"))
    now = datetime.fromisoformat("2026-11-07T03:00:00+00:00")
    result = SearchService(store, config, client=fake).search("지난달 사진", now=now)
    assert fake.seen_now.isoformat() == "2026-11-07T12:00:00+09:00"
    assert result["results"][0]["metadata"]["captured_at"].startswith("2026-10")


def test_invalid_model_date_requests_clarification(tmp_path):
    config = settings(tmp_path)
    store = MemoryStore(tmp_path)
    add_photo(store, config)
    fake = FakeAI(plan(captured_from="2026-13-99"))
    result = SearchService(store, config, client=fake).search("날짜가 이상한 사진")
    assert result["status"] == "needs_clarification"


def test_missing_index_is_configuration_error_not_no_match(tmp_path):
    config = settings(tmp_path)
    store = MemoryStore(tmp_path)
    add_photo(store, config, indexed=False)
    with pytest.raises(ConfigurationError, match="인덱싱"):
        SearchService(store, config, client=FakeAI()).search("사진")


def test_changed_model_cannot_search_stale_embeddings(tmp_path):
    config = settings(tmp_path)
    store = MemoryStore(tmp_path)
    add_photo(store, config)
    config.embedding_model = "different-model"
    with pytest.raises(ConfigurationError, match="인덱싱"):
        SearchService(store, config, client=FakeAI()).search("사진")


def test_index_restart_reuses_caption_after_embedding_failure(tmp_path):
    config = settings(tmp_path)
    store = MemoryStore(tmp_path)
    add_photo(store, config, indexed=False)
    fake = FakeAI()
    original_embed = fake.embed
    fake.embed = lambda text: (_ for _ in ()).throw(RuntimeError("temporary embedding failure"))
    first = index_gallery(store, config, client=fake)
    assert first["status"] == "partial" and fake.caption_calls == 1
    fake.embed = original_embed
    second = index_gallery(store, config, client=fake)
    third = index_gallery(store, config, client=fake)
    assert second["indexed"] == 1 and third["reused"] == 1
    assert fake.caption_calls == 1 and fake.embed_calls == 1


def test_concurrent_index_resume_preserves_exclusions_and_completed_work(tmp_path):
    config = settings(tmp_path)
    store = MemoryStore(tmp_path)
    for name in ["one", "two", "three", "four", "five", "rejected"]:
        photo = add_photo(store, config, name, indexed=False)
        if name == "rejected":
            photo["review"] = {"included": False, "reason": "무대 사진 제외"}
            store.upsert_photo(photo)

    class ConcurrentAI(FakeAI):
        def __init__(self):
            super().__init__()
            self.lock = Lock()
            self.barrier = Barrier(2)
            self.failed_once = False
            self.threads = set()

        def caption(self, photo):
            with self.lock:
                self.caption_calls += 1
                count = self.caption_calls
                self.threads.add(get_ident())
            if count <= 2:
                self.barrier.wait(timeout=5)
            return Caption(**{**CAPTION, "caption_ko": photo["photo_id"] + " 검은 옷"})

        def embed(self, text):
            with self.lock:
                self.embed_calls += 1
                if "two 검은 옷" in text and not self.failed_once:
                    self.failed_once = True
                    raise RuntimeError("temporary embedding failure")
            return np.array([1, 0, 0], dtype=np.float32)

    ai = ConcurrentAI()
    events = []
    first = index_gallery(store, config, client=ai, workers=2, progress=events.append)
    assert len(ai.threads) == 2
    assert (first["indexed"], first["excluded"], first["status"]) == (4, 1, "partial")
    assert first["failed"] == [{"photo_id": "two", "error": "temporary embedding failure"}]
    assert sorted(event["processed"] for event in events) == list(range(1, 7))
    assert all(event["total"] == 6 for event in events)
    second = index_gallery(store, config, client=ai, workers=2)
    assert (second["indexed"], second["reused"], second["excluded"], second["status"]) == (5, 4, 1, "ok")
    assert ai.caption_calls == 5 and ai.embed_calls == 6
    assert store.photos["rejected"]["exclusion_reason"] == "무대 사진 제외"


def test_concurrent_index_fatal_configuration_stops_remaining_gallery(tmp_path):
    config = settings(tmp_path)
    store = MemoryStore(tmp_path)
    for index in range(20):
        add_photo(store, config, str(index), indexed=False)
    ai = FakeAI()

    def fail(photo):
        ai.caption_calls += 1
        raise ConfigurationError("model not available")

    ai.caption = fail
    with pytest.raises(ConfigurationError, match="model not available"):
        index_gallery(store, config, client=ai, workers=3)
    assert 1 <= ai.caption_calls <= 3
    assert ai.embed_calls == 0


@pytest.mark.parametrize("workers", [0, 9, True, 1.5])
def test_index_rejects_invalid_worker_count(tmp_path, workers):
    with pytest.raises(ValueError, match="workers"):
        index_gallery(MemoryStore(tmp_path), settings(tmp_path), workers=workers)


def test_excluded_image_never_gets_embedding(tmp_path):
    config = settings(tmp_path)
    store = MemoryStore(tmp_path)
    add_photo(store, config, indexed=False)
    fake = FakeAI()
    fake.caption = lambda photo: Caption(**{**CAPTION, "contains_people": False})
    result = index_gallery(store, config, client=fake)
    assert result["excluded"] == 1 and result["indexed"] == 0 and fake.embed_calls == 0
    assert result["status"] == "ok"


def test_api_key_absent_errors_without_network(tmp_path):
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        GalleryAI(MemoryStore(tmp_path), settings(tmp_path)).embed("사진")


def test_openai_boundary_caches_and_records_actual_usage(tmp_path):
    config = settings(tmp_path)
    store = MemoryStore(tmp_path)
    photo = add_photo(store, config, indexed=False)
    Image.new("RGB", (30, 30), "red").save(tmp_path / photo["image_path"])
    calls = []

    def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_parsed=Caption(**CAPTION), status="completed",
                               usage={"input_tokens": 100, "output_tokens": 30})

    def embed(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0, 0.0])],
                               usage={"prompt_tokens": 20, "total_tokens": 20})

    sdk = SimpleNamespace(responses=SimpleNamespace(parse=parse), embeddings=SimpleNamespace(create=embed))
    ai = GalleryAI(store, config, client=sdk)
    assert ai.caption(photo) == ai.caption(photo)
    assert np.array_equal(ai.embed("노래"), ai.embed("노래"))
    assert len(calls) == 2 and len(store.usages) == 2
    assert calls[0]["reasoning"] == {"effort": "none"}
    assert calls[0]["store"] is False


def test_refusal_is_not_cached_and_usage_is_still_recorded(tmp_path):
    config = settings(tmp_path)
    store = MemoryStore(tmp_path)
    sdk = SimpleNamespace(responses=SimpleNamespace(parse=lambda **kwargs: SimpleNamespace(
        output_parsed=None, status="completed", usage={"input_tokens": 20, "output_tokens": 5}
    )))
    ai = GalleryAI(store, config, client=sdk)
    with pytest.raises(ModelResponseError):
        ai.plan_query("사진", datetime.fromisoformat("2026-11-07T12:00:00+09:00"), ["서울"])
    assert not store.cache and len(store.usages) == 1


def test_sdk_parse_validation_error_records_unknown_usage_and_resumes(tmp_path):
    config = settings(tmp_path)
    store = MemoryStore(tmp_path)
    photo = add_photo(store, config, indexed=False)
    Image.new("RGB", (30, 30), "red").save(tmp_path / photo["image_path"])
    calls = 0

    def parse(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return Caption.model_validate({"caption_ko": "PRIVATE RAW RESPONSE"})
        return SimpleNamespace(output_parsed=Caption(**CAPTION), status="completed",
                               usage={"input_tokens": 100, "output_tokens": 30})

    ai = GalleryAI(store, config, client=SimpleNamespace(responses=SimpleNamespace(parse=parse)))
    with pytest.raises(ModelResponseError, match="usage unavailable") as caught:
        ai.caption(photo)
    assert "PRIVATE RAW RESPONSE" not in str(caught.value)
    assert caught.value.__suppress_context__
    assert store.usages[0][0] == "caption_parse_error"
    assert store.usages[0][2] == {"usage_unavailable": True, "error_type": "ValidationError"}
    assert not store.cache
    assert ai.caption(photo) == ai.caption(photo)
    assert calls == 2 and len(store.usages) == 2


def test_search_uses_store_sql_candidates_without_reading_corpus_vectors(tmp_path):
    config = settings(tmp_path)
    store = MemoryStore(tmp_path)
    add_photo(store, config)
    fake = FakeAI(plan(semantic_query="노래"))
    result = SearchService(store, config, client=fake).search("노래하는 사진")
    assert result["results"][0]["photo_id"] == "one"
    assert store.embedding_reads == 0
    assert len(store.candidate_queries) == 2
    assert store.candidate_queries[0][0] is None
    assert store.candidate_queries[1][0] == [1.0, 0.0, 0.0]


def test_postgres_index_search_and_live_metadata_update_without_local_assets(tmp_path, pg_store_factory):
    """Exercise real Store + indexing + SQL retrieval, using only model calls as doubles."""
    store = pg_store_factory(tmp_path / "data")
    config = settings(store.data_dir)
    config.embedding_dimensions = 1536
    for photo_id, location, captured in [
        ("a", "서울", "2026-10-01T00:00:00+09:00"),
        ("b", "부산", "2026-10-01T00:00:00+09:00"),
        ("c", "서울", "2026-11-01T00:00:00+09:00"),
    ]:
        path = store.data_dir / f"{photo_id}.jpg"
        Image.new("RGB", (32, 24), "red").save(path)
        store.upsert_photo({
            "photo_id": photo_id, "gallery_id": "bts-demo", "image_path": path.name,
            "identities": ["Jimin"], "source": {"attribution": "Test image"},
            "metadata": {"origin": "synthetic", "location": location, "captured_at": captured},
        })
        path.unlink()  # Original BYTEA remains sufficient for subsequent operations.

    class DatabaseAI(FakeAI):
        def embed(self, text):
            self.embed_calls += 1
            vector = np.zeros(1536, dtype=np.float32)
            vector[0] = 1
            return vector

    ai = DatabaseAI(plan(
        semantic_query="노래", location="서울", people=["지민"],
        captured_from="2026-10-01T00:00:00+09:00", captured_until="2026-11-01T00:00:00+09:00",
    ))
    assert index_gallery(store, config, client=ai)["indexed"] == 3
    assert index_gallery(store, config, client=ai)["reused"] == 3
    service = SearchService(store, config, client=ai)
    result = service.search("10월 서울 지민 노래 사진")
    assert [item["photo_id"] for item in result["results"]] == ["a"]
    assert result["results"][0]["score"] == pytest.approx(1)
    photo = store.get_photo("a")
    photo["metadata"]["location"] = "부산"
    store.upsert_photo(photo)
    calls_before = ai.embed_calls
    assert service.search("10월 서울 지민 노래 사진")["status"] == "no_match"
    assert ai.embed_calls == calls_before
    assert not list(store.data_dir.glob("*.jpg"))
