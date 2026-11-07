"""Self-contained, escaped HTML previews; no web server or external assets."""
import base64
import json
from html import escape
from pathlib import Path

from .store import Store


def _text(value) -> str:
    if isinstance(value, dict):
        value = value.get("caption_ko", json.dumps(value, ensure_ascii=False))
    return escape(str(value or ""), quote=True)


def write_preview(store: Store, result: dict, output: Path | None = None) -> Path:
    cards = []
    for index, item in enumerate(result.get("results", []), 1):
        photo = store.get_photo(item["photo_id"], result["gallery_id"])
        if not photo:
            continue
        image_bytes, mime = store.get_asset(photo["photo_id"], result["gallery_id"], thumbnail=True)
        uri = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
        metadata = photo["metadata"]
        source = photo["source"]
        origin = "가상 메타데이터" if metadata.get("origin") == "synthetic" else "실제 메타데이터"
        cards.append(f'''<article><img src="{uri}" alt="후보 {index}">
          <h2>{index}. {_text(photo['photo_id'])}</h2>
          <p>{_text(item.get('caption') or photo.get('caption') or '아직 AI 설명이 없습니다.')}</p>
          <p class="badge">{origin}</p>
          <p>{_text(metadata.get('location'))} · {_text(metadata.get('captured_at'))}</p>
          <p>{_text(item.get('reason', ''))}</p>
          <p class="credit">{_text(source.get('attribution'))}<br>
          {_safe_link(source.get('page_url'), '원본 출처')} ·
          {_safe_link(source.get('license_url'), source.get('license_name', '라이선스'))}</p>
        </article>''')
    document = '''<!doctype html><html lang="ko"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>BTS Gallery</title>
    <style>body{margin:32px auto;padding:0 24px;max-width:1200px;font:16px/1.6 system-ui;color:#202026;background:#faf9f7}
    h1{font-size:26px} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:22px}
    article{background:white;border:1px solid #ddd;border-radius:12px;padding:16px;overflow-wrap:anywhere}
    img{width:100%;height:320px;object-fit:contain;background:#f1f1f1} h2{font-size:17px}
    .badge{color:#744200;background:#fff1d6;display:inline-block;padding:2px 8px;border-radius:4px}
    .credit{font-size:12px;color:#555}code{overflow-wrap:anywhere}</style>
    <h1>BTS 사진 검색</h1>'''
    document += f"<p>{_text(result.get('query', '공유 갤러리'))}</p>"
    document += f"<p>상태: {_text(result.get('status', 'ok'))} · 결과 ID: <code>{_text(result.get('result_id', 'gallery'))}</code></p>"
    if result.get("clarification"):
        document += f"<p>{_text(result['clarification'])}</p>"
    if not cards:
        document += "<p>표시할 사진이 없습니다.</p>"
    document += '<div class="grid">' + "".join(cards) + "</div></html>"
    output = output or store.data_dir / "previews" / f"{result.get('result_id', 'gallery')}.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output.resolve()


def _safe_link(url, label) -> str:
    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
        return _text(label)
    return f'<a href="{escape(url, quote=True)}" rel="noreferrer">{_text(label)}</a>'


def select_photo(store: Store, result_id: str, number: int, destination: str) -> dict:
    result = store.get_search(result_id)
    if not result:
        raise ValueError("검색 결과 ID가 없습니다. 먼저 gallery search를 실행하세요.")
    if result["status"] != "ok" or not 1 <= number <= len(result["results"]):
        raise ValueError("선택 번호는 해당 검색 결과에 표시된 번호여야 합니다.")
    if not destination.strip():
        raise ValueError("공유 대상 이름을 지정하세요.")
    selected = result["results"][number - 1]
    photo = store.get_photo(selected["photo_id"], result["gallery_id"])
    if not photo:
        raise ValueError("선택한 사진이 더 이상 갤러리에 없습니다.")
    store.get_asset(photo["photo_id"], result["gallery_id"])
    return store.save_selection({
        "status": "simulated", "sent": False, "transport": "mock", "result_id": result_id,
        "number": number, "photo_id": photo["photo_id"], "gallery_id": result["gallery_id"],
        "destination": destination.strip(), "image_storage": "postgresql:bytea",
        "attribution": photo["source"]["attribution"],
        "message": "모의 공유 결과입니다. Vooy에 전송하지 않았습니다.",
    })
