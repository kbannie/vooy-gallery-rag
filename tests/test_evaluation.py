import json

import pytest

from vooy_gallery.evaluation import evaluate, gallery_fingerprint


def test_evaluation_refuses_unreviewed_or_stale_labels(tmp_path, pg_store_factory):
    store = pg_store_factory(tmp_path / "data")
    dataset = tmp_path / "test.jsonl"
    case = {"id": "q1", "split": "test", "reviewed": False, "expected_status": "no_match"}
    dataset.write_text(json.dumps(case))
    with pytest.raises(ValueError, match="검수"):
        evaluate(store, None, dataset)
    case.update(reviewed=True, gallery_fingerprint="stale")
    dataset.write_text(json.dumps(case))
    with pytest.raises(ValueError, match="변경"):
        evaluate(store, None, dataset)


def test_evaluation_uses_independent_filters_not_model_claims(tmp_path, pg_store_factory):
    from PIL import Image
    store = pg_store_factory(tmp_path / "data")
    Image.new("RGB", (32, 32), "blue").save(store.data_dir / "p.jpg")
    store.upsert_photo({"photo_id": "p", "gallery_id": "bts-demo", "image_path": "p.jpg",
                        "metadata": {"location": "부산", "captured_at": "2026-09-01T12:00:00+09:00"}})
    case = {"id": "q1", "split": "test", "reviewed": True, "expected_status": "ok",
            "query": "서울 사진", "relevant_photo_ids": ["p"], "now": "2026-11-07T13:00:00+09:00",
            "expected_filters": {"location": "서울"},
            "gallery_fingerprint": gallery_fingerprint(store.list_photos())}
    dataset = tmp_path / "test.jsonl"
    dataset.write_text(json.dumps(case))

    class BadService:
        def search(self, *args, **kwargs):
            return store.save_search({"status": "ok", "gallery_id": "bts-demo", "results": [{"photo_id": "p"}],
                                      "applied_filters": {"location": "부산"}})

    report = evaluate(store, BadService(), dataset)
    assert report["filter_violations"] == 1
    assert report["acceptance_passed"] is False


def test_precision_at_three_keeps_fixed_denominator(tmp_path, pg_store_factory):
    from PIL import Image

    store = pg_store_factory(tmp_path / "data")
    Image.new("RGB", (32, 32), "blue").save(store.data_dir / "p.jpg")
    store.upsert_photo({"photo_id": "p", "gallery_id": "bts-demo", "image_path": "p.jpg",
                        "metadata": {"location": "서울", "captured_at": "2026-09-01T12:00:00+09:00"}})
    case = {"id": "q1", "split": "test", "reviewed": True, "expected_status": "ok",
            "query": "서울 사진", "relevant_photo_ids": ["p"], "now": "2026-11-07T13:00:00+09:00",
            "expected_filters": {"location": "서울"},
            "gallery_fingerprint": gallery_fingerprint(store.list_photos())}
    dataset = tmp_path / "test.jsonl"
    dataset.write_text(json.dumps(case))

    class OneResultService:
        def search(self, *args, **kwargs):
            store.record_usage("query_plan", "model", {"input_tokens": 1000}, .10)
            store.record_usage("caption_parse_error", "model",
                               {"usage_unavailable": True, "error_type": "ValidationError"}, .10, .50)
            return store.save_search({"status": "ok", "gallery_id": "bts-demo", "results": [{"photo_id": "p"}]})

    report = evaluate(store, OneResultService(), dataset)
    assert report["records"][0]["precision_at_3"] == pytest.approx(1 / 3)
    assert report["mean_precision_at_3"] == pytest.approx(1 / 3)
    assert report["mean_precision_at_returned"] == 1
    assert report["estimated_usd"] == pytest.approx(.0001)
    assert report["unknown_usage_calls"] == 1 and report["cost_is_subtotal"] is True
    assert "Global usage-ledger delta" in report["cost_scope"]
    assert "concurrent requests" in report["cost_scope"]
