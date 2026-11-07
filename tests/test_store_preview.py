from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vooy_gallery.preview import select_photo, write_preview
from vooy_gallery.settings import Settings


@pytest.fixture
def store(tmp_path, pg_store_factory):
    store = pg_store_factory(tmp_path / "data")
    (store.data_dir / "images").mkdir()
    for photo_id in ["one", "two"]:
        Image.new("RGB", (32, 32), "blue").save(store.data_dir / "images" / f"{photo_id}.jpg")
        store.upsert_photo({"photo_id": photo_id, "gallery_id": "bts-demo",
                            "image_path": f"images/{photo_id}.jpg", "mime_type": "image/jpeg",
                            "sha256": photo_id, "metadata": {"origin": "synthetic", "location": "서울",
                            "captured_at": "2026-10-03T12:00:00+09:00"},
                            "source": {"attribution": '<script>alert("x")</script>',
                                       "page_url": "javascript:alert(1)", "license_url": "https://example.com"}})
    return store


def test_selection_is_bound_to_saved_result_and_never_sent(store):
    first = store.save_search({"status": "ok", "gallery_id": "bts-demo", "results": [{"photo_id": "two"}]})
    store.save_search({"status": "ok", "gallery_id": "bts-demo", "results": [{"photo_id": "one"}]})
    selection = select_photo(store, first["result_id"], 1, "친구 단톡방")
    assert selection["photo_id"] == "two"
    assert selection["destination"] == "친구 단톡방"
    assert selection["sent"] is False
    assert selection["status"] == "simulated"
    with pytest.raises(ValueError):
        select_photo(store, first["result_id"], 2, "친구")


def test_invalid_selection_does_not_write_receipt(store):
    with pytest.raises(ValueError):
        select_photo(store, "unknown", 1, "친구")
    result = store.save_search({"status": "no_match", "gallery_id": "bts-demo", "results": []})
    with pytest.raises(ValueError):
        select_photo(store, result["result_id"], 1, "친구")
    with store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) AS total FROM selections").fetchone()["total"] == 0


def test_preview_contains_image_and_escapes_source_html(store):
    result = store.save_search({"status": "ok", "query": "<script>bad</script>",
                               "gallery_id": "bts-demo", "results": [{"photo_id": "one"}]})
    html = write_preview(store, result).read_text()
    assert "data:image/jpeg;base64," in html
    assert "가상 메타데이터" in html
    assert "<script>" not in html
    assert 'href="javascript:' not in html


def test_vector_and_cache_survive_restart(store, pg_store_factory):
    vector = np.zeros(1536, dtype=np.float32)
    vector[:3] = [1, 2, 3]
    store.set_embedding("one", vector, "v1", "test-model")
    store.set_cache("api-result", {"caption": "공연 사진"})
    reopened = pg_store_factory(store.data_dir)
    assert np.array_equal(reopened.get_embedding("one", "v1"), vector)
    assert reopened.get_embedding("one", "v2") is None
    assert reopened.get_cache("api-result")["caption"] == "공연 사진"


@pytest.mark.parametrize("vector", [[0, 0], [float("nan"), 1], [[1, 2]], []])
def test_invalid_vectors_are_rejected(store, vector):
    with pytest.raises(ValueError):
        store.set_embedding("one", vector, "k", "m")


def test_gallery_scope_and_path_escape(store, tmp_path):
    assert store.get_photo("one", "private") is None
    (tmp_path / "secret").write_text("not an image")
    with pytest.raises(ValueError):
        store.asset_path("../secret")


def test_costs_are_separate_from_cached_calls(store):
    import json
    store.record_usage("caption", "test", {"input_tokens": 1000, "output_tokens": 100}, .1, .5)
    assert store.usage_summary()["total_estimated_usd"] == pytest.approx(.00015)
    assert store.usage_summary()["groups"][0]["calls"] == 1
    assert isinstance(json.loads(json.dumps(store.usage_summary()))["groups"][0]["input_tokens"], int)


def test_settings_do_not_print_api_key():
    assert "private-test-secret" not in repr(Settings(data_dir=Path("data"), api_key="private-test-secret"))
