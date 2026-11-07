"""Real PostgreSQL/pgvector integration tests, isolated by per-test schemas.

Set TEST_DATABASE_URL to the dedicated gallery_test database to run these.
"""
import hashlib
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from vooy_gallery.store import EMBEDDING_DIMENSIONS, StoreError


def photo(store, photo_id="photo-a", location="서울", captured_at="2026-10-10T12:00:00+09:00",
          people=None, gallery_id="bts-demo"):
    source = BytesIO()
    Image.new("RGB", (16, 12), "red").save(source, format="PNG")
    data = source.getvalue()
    relative = f"images/{photo_id}.png"
    path = store.data_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    document = {
        "photo_id": photo_id, "gallery_id": gallery_id,
        "image_path": relative, "thumbnail_path": relative,
        "metadata": {"origin": "synthetic", "location": location, "captured_at": captured_at},
        "identities": people or [], "source": {"attribution": "Example / CC BY 4.0"},
    }
    store.upsert_photo(document)
    return document, data


def vector(first=1.0, second=0.0):
    result = np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)
    result[:2] = [first, second]
    return result


def test_bytes_metadata_and_vector_are_in_postgres(pg_store_factory, tmp_path):
    store = pg_store_factory(tmp_path / "cache")
    document, original = photo(store)
    store.set_embedding(document["photo_id"], vector(), "caption-model-v1", "text-embedding-3-small")
    with store.connect() as conn:
        row = conn.execute(
            "SELECT pg_typeof(image_bytes)::text AS binary_type,pg_typeof(document)::text AS metadata_type,"
            "image_bytes,thumbnail_bytes,document,sha256 FROM photos WHERE photo_id=%s", (document["photo_id"],)
        ).fetchone()
        dimensions = conn.execute("SELECT vector_dims(embedding) AS n FROM embeddings").fetchone()["n"]
    assert row["binary_type"] == "bytea"
    assert row["metadata_type"] == "jsonb"
    assert bytes(row["image_bytes"]) == original
    assert bytes(row["thumbnail_bytes"]) == original
    assert row["document"]["metadata"]["origin"] == "synthetic"
    assert row["sha256"] == hashlib.sha256(original).hexdigest()
    assert dimensions == 1536
    assert not (store.data_dir / "gallery.sqlite3").exists()


def test_cache_is_disposable_and_metadata_updates_preserve_bytes(pg_store_factory, tmp_path):
    store = pg_store_factory(tmp_path / "cache")
    document, original = photo(store)
    path = store.data_dir / document["image_path"]
    path.write_bytes(b"stale or corrupt local cache")
    document["caption"] = {"caption_ko": "무대에 서 있는 사람"}
    store.upsert_photo(document)
    assert store.get_asset(document["photo_id"])[0] == original
    path.unlink()
    saved = store.get_photo(document["photo_id"])
    assert not path.exists()
    assert saved["caption"] == document["caption"]
    assert store.get_asset(document["photo_id"])[0] == original
    store.restore_cached_assets(saved)
    assert path.read_bytes() == original
    path.unlink()
    assert store.asset_path(document["image_path"]).read_bytes() == original
    path.unlink()
    assert store.list_photos()[0]["photo_id"] == document["photo_id"]
    assert not path.exists()
    assert store.get_asset(document["photo_id"], thumbnail=True)[1] == "image/png"
    with pytest.raises(ValueError, match="does not exist"):
        store.get_asset(document["photo_id"], gallery_id="private")


def test_sql_cosine_ranking_metadata_bounds_and_cache_keys(pg_store_factory, tmp_path):
    store = pg_store_factory(tmp_path / "cache")
    specs = [
        ("a", "서울", "2026-09-30T15:00:00+00:00", ["RM", "Jin"], vector(0.6, 0.8)),
        ("b", "서울", "2026-10-10T12:00:00+09:00", ["RM"], vector(0.8, 0.6)),
        ("c", "부산", "2026-10-10T12:00:00+09:00", ["RM"], vector()),
        ("d", "서울", "2026-11-01T00:00:00+09:00", ["RM"], vector()),
        ("stale", "서울", "2026-10-10T12:00:00+09:00", ["RM"], vector()),
    ]
    keys = {}
    for photo_id, location, captured, people, embedding in specs:
        photo(store, photo_id, location, captured, people)
        keys[photo_id] = f"key-{photo_id}"
        store.set_embedding(photo_id, embedding, keys[photo_id], "text-embedding-3-small")
    keys["stale"] = "new-prompt-key"
    for path in (store.data_dir / "images").iterdir():
        path.unlink()
    filters = {"location": "서울", "captured_from": "2026-10-01T00:00:00+09:00",
               "captured_until": "2026-11-01T00:00:00+09:00", "people": ["RM"]}
    results, count = store.search_candidates(vector(), "bts-demo", filters, keys, limit=10)
    assert count == 2
    assert [item["photo"]["photo_id"] for item in results] == ["b", "a"]
    assert [item["score"] for item in results] == pytest.approx([0.8, 0.6])
    assert not list((store.data_dir / "images").iterdir())
    assert store.count_indexed("bts-demo", keys) == 4
    assert store.count_indexed("private", keys) == 0
    assert store.get_embedding("stale", "new-prompt-key") is None
    assert store.get_embedding("a", "key-a").shape == (1536,)
    all_people = dict(filters, people=["RM", "Jin"])
    results, count = store.search_candidates(vector(), "bts-demo", all_people, keys)
    assert count == 1
    assert results[0]["photo"]["photo_id"] == "a"
    assert store.search_candidates(vector(), "private", {}, keys) == ([], 0)


def test_sql_metadata_ordering_unknown_dates_and_parameters(pg_store_factory, tmp_path):
    store = pg_store_factory(tmp_path / "cache")
    keys = {}
    for photo_id, captured in [
        ("old", "2026-08-01T00:00:00+09:00"),
        ("new-b", "2026-10-10T00:00:00+09:00"),
        ("new-a", "2026-10-10T00:00:00+09:00"),
        ("unknown", "2026-10-11T00:00:00"),
    ]:
        photo(store, photo_id, captured_at=captured)
        keys[photo_id] = f"key-{photo_id}"
        store.set_embedding(photo_id, vector(), keys[photo_id], "test")
    results, count = store.search_candidates(None, "bts-demo", {}, keys)
    assert count == 4
    assert [row["photo"]["photo_id"] for row in results] == ["new-a", "new-b", "old", "unknown"]
    assert all(row["score"] is None for row in results)
    results, count = store.search_candidates(None, "bts-demo", {"captured_from": "2026-09-01T00:00:00+09:00"}, keys)
    assert count == 2
    assert len(results) == 2
    assert store.search_candidates(None, "bts-demo", {"location": "서울' OR 1=1 --"}, keys) == ([], 0)
    assert store.count_indexed("bts-demo", keys) == 4
    with pytest.raises(ValueError, match="timezone"):
        store.search_candidates(None, "bts-demo", {"captured_from": "2026-10-01"}, keys)


@pytest.mark.parametrize("bad_vector", [[1, 0, 0], [0.0] * 1536, [float("nan")] * 1536])
def test_vector_dimension_and_finite_validation(pg_store_factory, tmp_path, bad_vector):
    store = pg_store_factory(tmp_path / "cache")
    photo(store)
    with pytest.raises(ValueError, match="1536"):
        store.set_embedding("photo-a", bad_vector, "key", "model")
    with pytest.raises(ValueError, match="1536"):
        store.search_candidates(bad_vector, "bts-demo", {}, {"photo-a": "key"})


def test_cache_usage_search_sessions_and_transaction_rollback(pg_store_factory, tmp_path):
    store = pg_store_factory(tmp_path / "cache")
    store.set_cache("a", {"text": "공연 사진"})
    store.set_cache("a", {"text": "업데이트"})
    assert store.get_cache("a") == {"text": "업데이트"}
    assert store.get_cache("missing") is None
    store.record_usage("caption", "model", {"input_tokens": 1000, "output_tokens": 100}, 0.10, 0.50)
    assert store.usage_summary()["total_estimated_usd"] == pytest.approx(0.00015)
    result = store.save_search({"gallery_id": "bts-demo", "query": "사진", "results": []})
    assert store.get_search(result["result_id"]) == result
    assert store.get_search("missing") is None
    selection = store.save_selection({"result_id": result["result_id"], "delivery_status": "not_sent"})
    with store.connect() as conn:
        row = conn.execute("SELECT document FROM selections WHERE selection_id=%s", (selection["selection_id"],)).fetchone()
    assert row["document"] == selection
    with pytest.raises(ValueError, match="abort transaction"), store.connect() as conn:
        conn.execute("INSERT INTO cache (key,value) VALUES (%s,%s::jsonb)", ("rollback", "{}"))
        raise ValueError("abort transaction")
    assert store.get_cache("rollback") is None
    with pytest.raises(StoreError) as caught, store.connect() as conn:
        conn.execute("SELECT nonexistent_column FROM photos")
    assert "postgresql://" not in str(caught.value)


def test_usage_preserves_unknown_tokens_and_reports_known_subtotal(pg_store_factory, tmp_path):
    store = pg_store_factory(tmp_path / "cache")
    store.record_usage("caption", "model", {"input_tokens": 1000, "output_tokens": 100}, .10, .50)
    initial = store.usage_summary()
    assert initial["unknown_usage_calls"] == 0 and initial["total_is_subtotal"] is False
    store.record_usage("caption_parse_error", "model",
                       {"usage_unavailable": True, "error_type": "ValidationError"}, .10, .50)
    with store.connect() as conn:
        row = conn.execute("SELECT input_tokens,output_tokens,estimated_usd,details FROM usage "
                           "WHERE operation='caption_parse_error'").fetchone()
        migrations = conn.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
    assert [item["version"] for item in migrations] == [1, 2]
    assert row["input_tokens"] is row["output_tokens"] is row["estimated_usd"] is None
    assert row["details"] == {"usage_unavailable": True, "error_type": "ValidationError"}
    # Initialization is repeatable and does not erase unknown usage.
    store.initialize()
    summary = store.usage_summary()
    assert summary["unknown_usage_calls"] == 1 and summary["total_is_subtotal"] is True
    assert summary["total_estimated_usd"] == pytest.approx(.00015)
    group = next(item for item in summary["groups"] if item["operation"] == "caption_parse_error")
    assert group["input_tokens"] is group["output_tokens"] is group["estimated_usd"] is None
    assert group["calls"] == group["unknown_usage_calls"] == 1
    assert "subtotal excludes 1" in summary["cost_note"]


def test_upsert_blocks_path_traversal_and_gallery_reassignment(pg_store_factory, tmp_path):
    store = pg_store_factory(tmp_path / "cache")
    document, _ = photo(store)
    with pytest.raises(ValueError, match="another gallery"):
        store.upsert_photo(dict(document, gallery_id="private"))
    with pytest.raises(ValueError, match="outside"):
        store.upsert_photo(dict(document, image_path="../outside.png"))
    assert store.get_photo(document["photo_id"])["gallery_id"] == "bts-demo"
