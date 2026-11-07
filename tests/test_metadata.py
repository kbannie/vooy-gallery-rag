from datetime import datetime
from io import BytesIO

from PIL import Image

from vooy_gallery.metadata import exif_metadata, extract_exif, synthetic_metadata


def test_synthetic_reproducible_bounded_and_marked():
    first = synthetic_metadata("commons-123")
    assert first == synthetic_metadata("commons-123")
    assert first["origin"] == "synthetic"
    assert first["latitude"] is None
    for index in range(150):
        metadata = synthetic_metadata(str(index))
        stamp = datetime.fromisoformat(metadata["captured_at"])
        assert "2026-08-01" <= stamp.date().isoformat() <= "2026-11-07"
        assert stamp.utcoffset().total_seconds() == 9 * 3600


def test_missing_exif_is_not_fabricated():
    output = BytesIO()
    Image.new("RGB", (50, 50)).save(output, "JPEG")
    raw = extract_exif(output.getvalue())
    assert raw["datetime_original"] is None
    assert exif_metadata(raw)["captured_at"] is None
    assert exif_metadata(raw)["location"] is None
    assert exif_metadata(raw)["origin"] == "exif"


def test_exif_offset_and_unknown_timezone_are_preserved():
    metadata = exif_metadata({"datetime_original": "2021:05:06 07:08:09", "offset_time_original": "+09:00"})
    assert metadata["captured_at"] == "2021-05-06T07:08:09+09:00"
    unknown = exif_metadata({"datetime_original": "2021:05:06 07:08:09"})
    assert unknown["captured_at"] == "2021-05-06T07:08:09"
    assert exif_metadata({"datetime_original": "nonsense"})["captured_at"] is None


def test_exif_reader_does_not_mutate_file(tmp_path):
    path = tmp_path / "original.jpg"
    Image.new("RGB", (60, 60)).save(path, "JPEG")
    original = path.read_bytes()
    extract_exif(path)
    assert path.read_bytes() == original
