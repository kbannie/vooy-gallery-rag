#!/usr/bin/env python3
"""Evaluate reviewed video queries without changing labels or model prompts.

This is a small manually reviewed smoke evaluation, not a held-out benchmark.
Only queries/text and stored descriptions go to the search provider; this script
does not upload videos or frames, index content, or print connection credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from vooy_gallery.retrieval import canonical_location, parse_time
from vooy_gallery.settings import Settings
from vooy_gallery.store import Store
from vooy_gallery.video_ai import segment_caption_key, segment_embedding_key
from vooy_gallery.video_search import VideoSearchService
from vooy_gallery.video_store import VideoStore


def next_month(value: datetime) -> datetime:
    return value.replace(year=value.year + (value.month == 12), month=value.month % 12 + 1, day=1)


def expected_filters(case: dict) -> tuple[dict, str]:
    """Infer simple date phrases only; locations require explicit reviewed labels."""
    if "expected_filters" in case:
        raw = case["expected_filters"]
        return {"location": raw.get("location"), "people": raw.get("people", []),
                "captured_from": raw.get("captured_from", raw.get("date_from")),
                "captured_until": raw.get("captured_until", raw.get("date_to"))}, "reviewed_label"
    now = parse_time(case["now"])
    if now is None:
        raise ValueError("Every evaluation case needs a timezone-aware now")
    query = case["query"]
    filters = {}
    match = re.search(r"(?<!\d)(\d{4})년(?:\s*(\d{1,2})월)?(?:\s*(\d{1,2})일)?", query)
    if match:
        year, month, day = match.groups()
        start = datetime(int(year), int(month or 1), int(day or 1), tzinfo=now.tzinfo)
        if day:
            from datetime import timedelta

            end = start + timedelta(days=1)
        elif month:
            end = next_month(start)
        else:
            end = start.replace(year=start.year + 1)
        filters.update(captured_from=start.isoformat(), captured_until=end.isoformat())
    elif "지난달" in query:
        end = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        start = end.replace(year=end.year - (end.month == 1), month=12 if end.month == 1 else end.month - 1)
        filters.update(captured_from=start.isoformat(), captured_until=end.isoformat())
    # A scene such as '체육관에서' is not necessarily a geographic metadata label.
    # Independent location/identity constraints must be supplied by the reviewer.
    return filters, "deterministic_simple_date_parser"


def filter_violations(metadata: dict, identities: list[str], filters: dict) -> list[str]:
    violations = []
    if filters.get("location") and canonical_location(metadata.get("location") or "") != canonical_location(filters["location"]):
        violations.append("location")
    if filters.get("people") and not set(filters["people"]).issubset(identities):
        violations.append("people")
    for field, earlier in (("captured_from", True), ("captured_until", False)):
        if not filters.get(field):
            continue
        try:
            captured, bound = parse_time(metadata.get("captured_at")), parse_time(filters[field])
            if captured is None or bound is None or (captured < bound if earlier else captured >= bound):
                violations.append(field)
        except (ValueError, TypeError):
            violations.append(field)
    return violations


def validate_result(store: VideoStore, result: dict, case: dict, independent_filters: dict) -> dict:
    gallery = case["gallery_id"]
    segment_errors, metadata_errors, plan_errors = [], [], []
    if result.get("gallery_id") != gallery:
        segment_errors.append({"field": "result_gallery_id"})
    rows = result.get("results", [])
    if len(rows) > 3:
        segment_errors.append({"field": "result_limit"})
    if result.get("status") == "no_match" and rows:
        segment_errors.append({"field": "no_match_with_results"})
    applied = result.get("applied_filters") or {}
    for field in ("captured_from", "captured_until", "location", "people"):
        expected = independent_filters.get(field)
        if not expected:
            continue
        actual = applied.get(field)
        try:
            equal = parse_time(actual) == parse_time(expected) if field.startswith("captured_") else actual == expected
        except (ValueError, TypeError):
            equal = False
        if not equal:
            plan_errors.append({"field": field, "expected": expected, "actual": actual})
    for row in rows:
        sid, vid = row.get("segment_id"), row.get("video_id")
        segment = store.get_segment(sid, gallery) if isinstance(sid, str) else None
        video = store.get_video(vid, gallery) if isinstance(vid, str) else None
        if segment is None or video is None or segment.get("video_id") != vid:
            segment_errors.append({"segment_id": sid, "field": "video_segment_membership"})
            continue
        start, end = row.get("start_ms"), row.get("end_ms")
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= video["duration_ms"]
                or start != segment["start_ms"] or end != segment["end_ms"]):
            segment_errors.append({"segment_id": sid, "field": "segment_bounds"})
        if row.get("frame_timestamps_ms") != segment["frame_timestamps_ms"]:
            segment_errors.append({"segment_id": sid, "field": "sample_timestamps"})
        for origin, filters in (("independent_expected_filters", independent_filters), ("model_applied_filters", applied)):
            for field in filter_violations(video.get("metadata") or {}, segment.get("identities", []), filters):
                metadata_errors.append({"segment_id": sid, "field": field, "constraint_source": origin})
    return {"segment_violations": segment_errors, "filter_violations": metadata_errors,
            "query_plan_violations": plan_errors}


def evaluate(store: VideoStore, settings: Settings, cases: list[dict], dataset_sha256: str,
             output: Path) -> dict:
    ids = [case["id"] for case in cases]
    if not cases or len(set(ids)) != len(ids):
        raise ValueError("Dataset must contain unique nonempty cases")
    expectations = {}
    required = set()
    for case in cases:
        if case.get("gallery_id") != settings.gallery_id or not case.get("review_basis"):
            raise ValueError("Each case needs the configured gallery and explicit manual review evidence")
        if case.get("expected_status") not in {"ok", "no_match"}:
            raise ValueError("This smoke evaluator supports ok/no_match labels")
        relevant = case.get("relevant_video_ids", [])
        if case["expected_status"] == "ok" and not relevant:
            raise ValueError("Positive cases need reviewed relevant_video_ids")
        if case["expected_status"] == "no_match" and relevant:
            raise ValueError("Negative cases must not contain positive video IDs")
        required.update(relevant)
        expectations[case["id"]] = expected_filters(case)
    videos = {video["video_id"]: video for video in store.list_videos(settings.gallery_id)}
    if not required.issubset(videos):
        raise ValueError("A reviewed video is missing from the configured gallery")
    segments = store.list_segments(settings.gallery_id)
    eligible = {segment["segment_id"]: segment_embedding_key(segment, settings) for segment in segments
                if segment.get("index_status") == "indexed" and segment.get("caption")
                and segment.get("caption_cache_key") == segment_caption_key(segment, settings)}
    expected_segments = [segment for segment in segments if segment["video_id"] in required]
    required_keys = {segment["segment_id"]: eligible[segment["segment_id"]]
                     for segment in expected_segments if segment["segment_id"] in eligible}
    if (not expected_segments or len(required_keys) != len(expected_segments)
            or store.count_indexed(settings.gallery_id, required_keys) != len(expected_segments)):
        raise ValueError("Required video segments are not fully indexed for the current model settings")
    before = store.base.usage_summary()
    service = VideoSearchService(store, settings)
    records = []
    report = {"evaluation_type": "manually_reviewed_three_video_smoke_test_not_held_out",
              "evaluator_version": "video-smoke-v1-date-only",
              "dataset_sha256": dataset_sha256, "gallery_id": settings.gallery_id,
              "vision_model": settings.vision_model, "embedding_model": settings.embedding_model,
              "video_count": len(videos), "indexed_segments": store.count_indexed(settings.gallery_id, eligible),
              "evidence_scope": "sampled_frames_only_not_continuous_timeline",
              "labels_and_prompts_changed": False, "records": records, "status": "running"}
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    for case in cases:
        started = time.perf_counter()
        filters, expectation_source = expectations[case["id"]]
        record = {"id": case["id"], "query": case["query"], "expected_status": case["expected_status"],
                  "relevant_video_ids": case["relevant_video_ids"], "review_basis": case["review_basis"],
                  "expected_filters": filters, "filter_expectation_source": expectation_source}
        try:
            result = service.search(case["query"], gallery_id=case["gallery_id"], limit=3,
                                    now=datetime.fromisoformat(case["now"]))
            returned = [row["video_id"] for row in result["results"][:3]]
            record.update(actual_status=result["status"], returned_video_ids=returned,
                          returned_segments=[{key: row[key] for key in ("video_id", "segment_id", "start_ms", "end_ms")}
                                             for row in result["results"]],
                          result_id=result["result_id"], applied_filters=result.get("applied_filters", {}),
                          semantic_query=result.get("semantic_query", ""),
                          hit_at_3=bool(set(returned) & set(case["relevant_video_ids"])),
                          correct_no_match=result["status"] == "no_match" and not result["results"],
                          **validate_result(store, result, case, filters))
        except Exception as exc:  # noqa: BLE001 -- save safe failure evidence, never provider bodies/secrets.
            record.update(actual_status="error", error_type=type(exc).__name__, hit_at_3=False,
                          correct_no_match=False, returned_video_ids=[], segment_violations=[],
                          filter_violations=[], query_plan_violations=[])
        record["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        records.append(record)
        save()
        print(json.dumps({"case_id": case["id"], "status": record["actual_status"],
                          "hit_at_3": record["hit_at_3"]}, ensure_ascii=False), flush=True)
    after = store.base.usage_summary()
    positive = [record for record in records if record["expected_status"] == "ok"]
    negative = [record for record in records if record["expected_status"] == "no_match"]
    hits = sum(record["hit_at_3"] for record in positive)
    negatives = sum(record["correct_no_match"] for record in negative)
    errors = sum(record["actual_status"] == "error" for record in records)
    violations = {key: sum(len(record[key]) for record in records)
                  for key in ("segment_violations", "filter_violations", "query_plan_violations")}
    unknown = after.get("unknown_usage_calls", 0) - before.get("unknown_usage_calls", 0)
    report.update(status="complete" if not errors else "partial", cases=len(records), positive_cases=len(positive),
                  hits_at_3=hits, hit_rate_at_3=hits / len(positive) if positive else None,
                  no_match_cases=len(negative), correct_no_match=negatives, errors=errors, **violations,
                  smoke_passed=hits == len(positive) and negatives == len(negative) and not errors and not any(violations.values()),
                  estimated_usd=after["total_estimated_usd"] - before["total_estimated_usd"],
                  unknown_usage_calls=unknown, cost_is_subtotal=unknown > 0,
                  cost_scope="Global ledger delta during evaluation; may include concurrent requests and excludes unknown usage.")
    save()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", type=Path, default=Path("eval/local_video_queries.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/video-eval-report.json"))
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    dataset = args.dataset if args.dataset.is_absolute() else root / args.dataset
    output = args.output if args.output.is_absolute() else root / args.output
    try:
        payload = dataset.read_bytes()
        cases = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
        settings = Settings.from_env(root / ".env")
        data_dir = Path(os.getenv("GALLERY_DATA_DIR", "data")).expanduser()
        settings = replace(settings, gallery_id=settings.video_gallery_id,
                           data_dir=data_dir if data_dir.is_absolute() else root / data_dir)
        store = VideoStore(Store(settings.data_dir, database_url=settings.database_url))
        report = evaluate(store, settings, cases, hashlib.sha256(payload).hexdigest(), output)
        print(json.dumps({key: report[key] for key in ("status", "cases", "hits_at_3", "positive_cases", "correct_no_match",
                                                        "no_match_cases", "segment_violations", "filter_violations",
                                                        "query_plan_violations", "smoke_passed", "estimated_usd")}, ensure_ascii=False))
        return 0 if report["smoke_passed"] else 2
    except Exception as exc:  # noqa: BLE001 -- do not expose DSNs, keys or provider payloads.
        print(json.dumps({"status": "error", "error_type": type(exc).__name__,
                          "message": "Video evaluation could not start or finish; check dataset and indexed gallery."}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
