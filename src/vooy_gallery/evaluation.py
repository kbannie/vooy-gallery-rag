"""Evaluate only explicitly reviewed, image-grounded labels, never self-generated answers."""
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path


def gallery_fingerprint(photos: list[dict]) -> str:
    manifest = [{"photo_id": p["photo_id"], "sha256": p["sha256"], "metadata": p["metadata"],
                 "identities": sorted(p.get("identities", []))}
                for p in sorted(photos, key=lambda p: p["photo_id"])]
    return hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def evaluate(store, service, dataset: Path, split: str = "test") -> dict:
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = [c for c in cases if c["split"] == split]
    if not cases:
        raise ValueError(f"평가 질문이 없습니다: split={split}")
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("평가 ID가 중복됩니다.")
    galleries = {c.get("gallery_id", "bts-demo") for c in cases}
    fingerprints = {g: gallery_fingerprint(store.list_photos(g)) for g in galleries}
    gallery_sizes = {g: sum(p.get("index_status") != "excluded" for p in store.list_photos(g)) for g in galleries}
    for case in cases:
        if not case.get("reviewed"):
            raise ValueError(f"이미지 검수가 완료되지 않은 평가 질문: {case['id']}")
        if case.get("gallery_fingerprint") != fingerprints[case.get("gallery_id", "bts-demo")]:
            raise ValueError("갤러리가 평가셋 작성 이후 변경되었습니다. 라벨을 다시 검수하세요.")
        expected = case["expected_status"]
        if expected not in {"ok", "no_match", "needs_clarification"}:
            raise ValueError("지원하지 않는 expected_status")
        if expected == "ok" and not case.get("relevant_photo_ids"):
            raise ValueError("정답 있는 질문에는 검수된 photo ID가 필요합니다.")
        for photo_id in case.get("relevant_photo_ids", []):
            if not store.get_photo(photo_id, case.get("gallery_id", "bts-demo")):
                raise ValueError(f"없는 정답 사진 ID: {photo_id}")
    usage_before = store.usage_summary()
    records = []
    for case in cases:
        started = time.perf_counter()
        result = service.search(case["query"], gallery_id=case.get("gallery_id", "bts-demo"),
                                limit=3, now=datetime.fromisoformat(case["now"]))
        returned = [r["photo_id"] for r in result["results"]]
        relevant = set(case.get("relevant_photo_ids", []))
        violations = []
        for photo_id in returned:
            photo = store.get_photo(photo_id, result["gallery_id"])
            meta = photo["metadata"]
            constraints = case.get("expected_filters", {})
            if constraints.get("location") and meta.get("location") != constraints["location"]:
                violations.append({"photo_id": photo_id, "field": "location"})
            if constraints.get("people") and not set(constraints["people"]).issubset(photo.get("identities", [])):
                violations.append({"photo_id": photo_id, "field": "people"})
            taken = meta.get("captured_at")
            for field, bad in (
                ("date_from", lambda t, b: t < b), ("date_to", lambda t, b: t >= b),
            ):
                bound = constraints.get(field)
                if bound and (not taken or bad(datetime.fromisoformat(taken), datetime.fromisoformat(bound))):
                    violations.append({"photo_id": photo_id, "field": field})
        records.append({"id": case["id"], "query": case["query"], "expected_status": case["expected_status"],
                        "actual_status": result["status"], "returned_photo_ids": returned,
                        "hit_at_3": bool(set(returned) & relevant),
                        "precision_at_3": len(set(returned) & relevant) / 3,
                        "precision_at_returned": len(set(returned) & relevant) / len(returned) if returned else 0,
                        "filter_violations": violations, "latency_seconds": time.perf_counter() - started,
                        "indexed_count": result.get("indexed_count", 0),
                        "result_id": result["result_id"]})
    positive = [r for r in records if r["expected_status"] == "ok"]
    negative = [r for r in records if r["expected_status"] == "no_match"]
    hits = sum(r["hit_at_3"] for r in positive)
    correct_no_match = sum(r["actual_status"] == "no_match" and not r["returned_photo_ids"] for r in negative)
    violations = sum(len(r["filter_violations"]) for r in records)
    complete_target = split == "test" and len(positive) == 25 and len(negative) == 5 and len(records) == 30
    complete_gallery = all(size >= 100 for size in gallery_sizes.values()) and all(r["indexed_count"] >= 100 for r in records)
    usage_after = store.usage_summary()
    unknown_usage_calls = usage_after.get("unknown_usage_calls", 0) - usage_before.get("unknown_usage_calls", 0)
    return {"split": split, "cases": len(records), "positive_cases": len(positive), "hits_at_3": hits,
            "hit_rate_at_3": hits / len(positive) if positive else None,
            "mean_precision_at_3": sum(r["precision_at_3"] for r in positive) / len(positive) if positive else None,
            "mean_precision_at_returned": sum(r["precision_at_returned"] for r in positive) / len(positive) if positive else None,
            "no_match_cases": len(negative), "correct_no_match": correct_no_match,
            "filter_violations": violations, "meets_dataset_size": complete_target,
            "gallery_sizes": gallery_sizes, "meets_gallery_size": complete_gallery,
            "acceptance_passed": complete_target and complete_gallery and hits >= 20 and correct_no_match == 5 and violations == 0,
            "estimated_usd": usage_after["total_estimated_usd"] - usage_before["total_estimated_usd"],
            "unknown_usage_calls": unknown_usage_calls,
            "cost_is_subtotal": unknown_usage_calls > 0,
            "cost_scope": "Global usage-ledger delta during this run; may include concurrent requests "
                          "and excludes calls with unavailable usage. Not isolated per-evaluation billing.",
            "records": records}
