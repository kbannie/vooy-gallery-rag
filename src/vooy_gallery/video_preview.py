"""Export a disposable local playback preview; PostgreSQL is the source of truth."""
import base64
import html
import json


def write_video_preview(video_store, settings):
    target = settings.data_dir / "previews" / "videos"
    target.mkdir(parents=True, exist_ok=True)
    cards = []
    segments = video_store.list_segments(settings.gallery_id)
    for video in video_store.list_videos(settings.gallery_id):
        vid = video["video_id"]
        data, _ = video_store.get_video_asset(vid, settings.gallery_id, "playback")
        filename = vid + ".mp4"
        (target / filename).write_bytes(data)
        entries = []
        for segment in (s for s in segments if s["video_id"] == vid):
            poster = base64.b64encode(video_store.get_segment_poster(segment["segment_id"], settings.gallery_id)).decode()
            caption = (segment.get("caption") or {}).get("caption_ko", "아직 설명을 생성하지 않았습니다.")
            entries.append(f'<img alt="구간 대표 프레임" src="data:image/jpeg;base64,{poster}">'
                           f'<p>{segment["start_ms"] / 1000:.2f}–{segment["end_ms"] / 1000:.2f}초: '
                           f'{html.escape(caption)}</p>')
        metadata = html.escape(json.dumps(video.get("metadata", {}), ensure_ascii=False))
        cards.append(f'<article><h2>{html.escape(vid)}</h2><video controls preload="metadata" '
                     f'src="{html.escape(filename)}"></video><p>{metadata}</p>{"".join(entries)}</article>')
    output = target / "index.html"
    output.write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>영상 갤러리</title>'
                      '<style>body{font:16px system-ui;margin:32px;background:#f4f5f7}article{background:white;'
                      'padding:24px;margin:20px 0;border-radius:16px}video{max-width:100%;max-height:440px}'
                      'img{max-height:180px;max-width:100%}p{overflow-wrap:anywhere}</style>'
                      '<h1>영상 갤러리 · 로컬 미리보기</h1><p>설명은 샘플 프레임 기준입니다. '
                      '재생 시간과 촬영 날짜는 서로 다른 정보입니다.</p>' + ''.join(cards) + '</html>',
                      encoding="utf-8")
    return output
