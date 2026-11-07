"""Export numbered contact sheets and a matching manifest for visual review."""
import argparse
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from vooy_gallery.settings import Settings
from vooy_gallery.store import Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery")
    parser.add_argument("--include-excluded", action="store_true")
    parser.add_argument("--unreviewed", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = Settings.from_env()
    store = Store(settings.data_dir, settings.database_url)
    gallery_id = args.gallery or settings.gallery_id
    photos = store.list_photos(gallery_id)
    if not args.include_excluded:
        photos = [p for p in photos if p.get("index_status") != "excluded"]
    if args.unreviewed:
        photos = [p for p in photos if "included" not in p.get("review", {})]
    target = args.output or Path(settings.data_dir) / "review" / gallery_id
    target.mkdir(parents=True, exist_ok=True)
    manifest = []
    for start in range(0, len(photos), 20):
        batch = photos[start:start + 20]
        sheet = Image.new("RGB", (1500, 1240), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for offset, photo in enumerate(batch):
            x, y = (offset % 5) * 300, (offset // 5) * 310
            data, _ = store.get_asset(photo["photo_id"], gallery_id, thumbnail=True)
            with Image.open(io.BytesIO(data)) as image:
                image = ImageOps.contain(image.convert("RGB"), (292, 265))
                sheet.paste(image, (x + (300 - image.width) // 2, y))
            number = start + offset + 1
            draw.text((x + 5, y + 268), f"{number:03d} {photo['photo_id']}", fill="black")
            draw.text((x + 5, y + 286), ", ".join(photo.get("identities", [])), fill="black")
            manifest.append({"number": number, **{k: v for k, v in photo.items() if k != "caption"}})
        sheet.save(target / f"sheet-{start // 20 + 1:02d}.jpg", quality=92)
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"photos": len(photos), "directory": str(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
