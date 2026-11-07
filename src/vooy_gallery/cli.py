import argparse
import json
import sys
import webbrowser
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .preview import select_photo, write_preview
from .settings import Settings
from .store import Store


def output(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="BTS 사진 검색 RAG / Vooy MCP 준비")
    parser.add_argument("--data-dir", type=Path, help="로컬 데이터 디렉터리")
    parser.add_argument("--env-file", default=".env")
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect", help="웹 검색으로 선정한 일상 사진 수집·가상 메타데이터 생성")
    collect.add_argument("--source", choices=["web", "commons"], default="web")
    collect.add_argument("--sources", type=Path, default=Path("config/daily_sources.json"))
    collect.add_argument("--target", type=int, default=100)
    collect.add_argument("--max-candidates", type=int, default=2000)
    index = sub.add_parser("index", help="OpenAI 사진 설명·임베딩 생성, 완료된 항목 재사용")
    index.add_argument("--limit", type=int)
    index.add_argument("--workers", type=int, default=4, help="동시에 처리할 사진 수 (1~8)")
    search = sub.add_parser("search", help="질문 검색 후 후보 이미지 미리보기 생성")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=3)
    search.add_argument("--now", help="재현 가능한 상대 날짜 기준 ISO8601 시각")
    search.add_argument("--open", action="store_true", help="미리보기를 기본 브라우저에서 열기")
    select = sub.add_parser("select", help="검색 결과에서 사진을 선택해 공유를 시뮬레이션")
    select.add_argument("result_id")
    select.add_argument("--number", type=int, required=True)
    select.add_argument("--to", required=True, dest="destination")
    preview = sub.add_parser("preview", help="수집된 전체 갤러리 미리보기")
    preview.add_argument("--open", action="store_true")
    sub.add_parser("status", help="사진·인덱스·키 설정 여부")
    sub.add_parser("db-init", help="PostgreSQL 스키마와 pgvector 확장 초기화")
    sub.add_parser("costs", help="단계별 API 토큰 사용량과 추정 비용")
    serve = sub.add_parser("serve", help="로컬 MCP 서버")
    serve.add_argument("--transport", choices=["streamable-http", "stdio"], default="streamable-http")
    video_import = sub.add_parser("video-import", help="로컬 MP4/MOV를 PostgreSQL에 저장하고 프레임 추출")
    video_import.add_argument("path", type=Path, help="영상 파일 또는 폴더")
    video_index = sub.add_parser("video-index", help="영상 구간 설명·pgvector 임베딩 생성")
    video_index.add_argument("--limit", type=int, help="처리할 영상 수")
    video_search = sub.add_parser("video-search", help="영상 장면 검색")
    video_search.add_argument("query")
    video_search.add_argument("--limit", type=int, default=3)
    video_search.add_argument("--now", help="상대 날짜 해석 기준, 시간대 포함 ISO8601")
    sub.add_parser("video-status", help="영상·구간·인덱스 현황")
    sub.add_parser("video-preview", help="브라우저 재생과 프레임을 확인할 로컬 HTML 생성")
    evaluate = sub.add_parser("evaluate", help="검수된 JSONL 평가셋 실행")
    evaluate.add_argument("dataset", type=Path)
    evaluate.add_argument("--split", choices=["dev", "test"], default="test")
    args = parser.parse_args(argv)
    try:
        settings = Settings.from_env(args.env_file)
        if settings.embedding_dimensions != 1536:
            raise ValueError("현재 pgvector 스키마는 vector(1536)입니다. EMBEDDING_DIMENSIONS=1536을 사용하세요.")
        if args.data_dir:
            settings = replace(settings, data_dir=args.data_dir.expanduser().resolve())
        store = Store(settings.data_dir, database_url=settings.database_url)
        if args.command.startswith("video-"):
            from .video_store import VideoStore

            video_settings = replace(settings, gallery_id=settings.video_gallery_id)
            videos = VideoStore(store)
            if args.command == "video-import":
                from .video_ingest import import_video

                source = args.path.expanduser().resolve()
                paths = ([source] if source.is_file() else sorted(
                    path for path in source.iterdir() if path.is_file() and path.suffix.lower() in {".mp4", ".mov"}
                ))
                if not paths:
                    raise ValueError("선택한 경로에 MP4/MOV 영상이 없습니다.")
                result = {"status": "ok", "videos": [], "failed": []}
                for path in paths:
                    try:
                        video = import_video(videos, video_settings, path)
                        result["videos"].append({"file": path.name, "video_id": video["video_id"],
                                                 "duration_ms": video["duration_ms"]})
                    except (ValueError, RuntimeError, OSError) as exc:
                        result["status"] = "partial"
                        result["failed"].append({"file": path.name, "error_type": type(exc).__name__,
                                                 "message": str(exc)})
                output(result)
                return 0 if not result["failed"] else 2
            if args.command == "video-index":
                from .video_search import index_videos

                result = index_videos(videos, video_settings, limit=args.limit)
                output(result)
                return 0 if result["status"] == "ok" else 2
            if args.command == "video-search":
                from .video_search import VideoSearchService

                now = datetime.fromisoformat(args.now) if args.now else None
                output(VideoSearchService(videos, video_settings).search(args.query, limit=args.limit, now=now))
            elif args.command == "video-preview":
                from .video_preview import write_video_preview

                output({"preview_file": str(write_video_preview(videos, video_settings))})
            else:
                segments = videos.list_segments(video_settings.gallery_id)
                output({"gallery_id": video_settings.gallery_id,
                        "videos": len(videos.list_videos(video_settings.gallery_id)), "segments": len(segments),
                        "indexed_segments": sum(s.get("index_status") == "indexed" for s in segments)})
            return 0
        if args.command == "collect":
            if args.target < 1 or args.max_candidates < 1:
                raise ValueError("target과 max-candidates는 양수여야 합니다.")
            def progress(event):
                print(json.dumps(event, ensure_ascii=False, default=str), file=sys.stderr, flush=True)
            if args.source == "web":
                from .web_collector import collect_web
                result = collect_web(store, args.sources, target=args.target, gallery_id=settings.gallery_id,
                                     progress=progress)
            else:
                from .collector import collect
                result = collect(store, target=args.target, gallery_id=settings.gallery_id,
                                 max_candidates=args.max_candidates, progress=progress)
            output(result)
            return 0 if result.get("status") == "target_reached" else 2
        if args.command == "db-init":
            output({"status": "ready", "storage": "PostgreSQL BYTEA + pgvector", "gallery_id": settings.gallery_id})
        elif args.command == "index":
            from .retrieval import index_gallery
            if args.limit is not None and args.limit < 1:
                raise ValueError("limit는 양수여야 합니다.")
            def index_progress(event):
                print(json.dumps(event, ensure_ascii=False, default=str), file=sys.stderr, flush=True)
            result = index_gallery(store, settings, limit=args.limit, workers=args.workers, progress=index_progress)
            output(result)
            if result.get("status") != "ok":
                return 2
        elif args.command == "search":
            from .retrieval import SearchService
            now = datetime.fromisoformat(args.now) if args.now else None
            if now is not None and now.tzinfo is None:
                raise ValueError("--now에 시간대를 포함하세요. 예: 2026-11-07T13:00:00+09:00")
            result = SearchService(store, settings).search(args.query, gallery_id=settings.gallery_id,
                                                         limit=args.limit, now=now)
            path = write_preview(store, result)
            output(dict(result, preview_file=str(path)))
            if args.open:
                webbrowser.open(path.as_uri())
        elif args.command == "select":
            result = select_photo(store, args.result_id, args.number, args.destination)
            target = store.data_dir / "shares" / f"{result['selection_id']}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            output(dict(result, receipt_file=str(target)))
        elif args.command == "preview":
            photos = [p for p in store.list_photos(settings.gallery_id) if p.get("index_status") != "excluded"]
            result = {"gallery_id": settings.gallery_id, "result_id": "gallery", "query": "전체 갤러리",
                      "status": "ok", "results": [{"photo_id": p["photo_id"]} for p in photos]}
            path = write_preview(store, result)
            output({"photos": len(photos), "preview_file": str(path)})
            if args.open:
                webbrowser.open(path.as_uri())
        elif args.command == "status":
            photos = store.list_photos(settings.gallery_id)
            from .ai import caption_key, embedding_key
            from .evaluation import gallery_fingerprint
            eligible = {p["photo_id"]: embedding_key(p, settings) for p in photos
                        if p.get("index_status") == "indexed" and p.get("caption_cache_key") == caption_key(p, settings)}
            output({"gallery_id": settings.gallery_id, "data_dir": str(settings.data_dir), "photos": len(photos),
                    "usable_photos": sum(p.get("index_status") != "excluded" for p in photos),
                    "excluded_photos": sum(p.get("index_status") == "excluded" for p in photos),
                    "captioned": sum(bool(p.get("caption")) for p in photos),
                    "embedded": store.count_indexed(settings.gallery_id, eligible),
                    "openai_key_configured": bool(settings.api_key), "vision_model": settings.vision_model,
                    "embedding_model": settings.embedding_model, "gallery_fingerprint": gallery_fingerprint(photos)})
        elif args.command == "costs":
            output(store.usage_summary())
        elif args.command == "serve":
            from .mcp_server import build_server
            build_server(settings, store=store).run(transport=args.transport)
        elif args.command == "evaluate":
            from .evaluation import evaluate
            from .retrieval import SearchService
            report = evaluate(store, SearchService(store, settings), args.dataset, args.split)
            target = store.data_dir / "evaluations" / f"{args.split}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            output(dict(report, report_file=str(target)))
            return 0 if report["acceptance_passed"] or args.split == "dev" else 2
        return 0
    except KeyboardInterrupt:
        print("중단했습니다. 완료된 데이터는 저장되어 있어 재실행할 수 있습니다.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 -- CLI boundary provides a concise failure without a traceback.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
