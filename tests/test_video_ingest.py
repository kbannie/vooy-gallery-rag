import hashlib
import io
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from vooy_gallery import video_ingest
from vooy_gallery.video_ingest import (
    MAX_VIDEO_BYTES,
    VIDEO_IMPORT_VERSION,
    VideoImportError,
    embedded_metadata,
    import_video,
    segment_plan,
)


class MemoryVideos:
    def __init__(self):
        self.documents = {}
        self.assets = {}

    def get_video(self, video_id, gallery_id):
        return self.documents.get((gallery_id, video_id))

    def put_video(self, document, original_bytes, playback_bytes, segments):
        key = (document["gallery_id"], document["video_id"])
        self.documents[key] = document
        self.assets[key] = (original_bytes, playback_bytes, segments)
        return document


def test_segment_coverage_and_bounded_last_midpoint():
    plans = segment_plan(25_001)
    assert [(p["start_ms"], p["end_ms"]) for p in plans] == [(0, 10_000), (10_000, 20_000), (20_000, 25_001)]
    assert [p["frame_timestamps_ms"] for p in plans] == [[0, 5000], [10_000, 15_000], [20_000, 22_500]]
    assert segment_plan(1)[0]["frame_timestamps_ms"] == [0, 0]
    assert len(segment_plan(120_000)) == 12
    with pytest.raises(VideoImportError):
        segment_plan(120_001)


def test_embedded_time_and_gps_without_city_inference():
    actual = embedded_metadata({"creation_time": "2025-08-20T01:02:03Z",
                                "com.apple.quicktime.creationdate": "2025-08-19T16:02:03-0900",
                                "com.apple.quicktime.location.ISO6709": "+37.5000+127.2500+012.000/"})
    assert actual["captured_at"] == "2025-08-19T16:02:03-09:00"
    assert actual["origin"] == "embedded"
    assert actual["latitude"] == 37.5
    assert actual["longitude"] == 127.25
    assert actual["location"] is None
    unknown = embedded_metadata({"creation_time": "2025-08-20T01:02:03", "location": "+99.5+999.0/"})
    assert unknown["captured_at"] is None
    assert unknown["origin"] == "unknown"
    assert unknown["latitude"] is None
    # A timezone-aware encode/export timestamp is still not evidence of capture.
    exported = embedded_metadata({"creation_time": "2026-11-07T01:02:03Z"})
    assert exported["captured_at"] is None
    assert exported["origin"] == "unknown"


@pytest.mark.parametrize("name", ["missing.mov", "clip.avi", "https:/example.org/video.mp4"])
def test_local_file_and_extension_validation(tmp_path, name):
    path = Path(name) if name.startswith("https:") else tmp_path / name
    with pytest.raises(VideoImportError):
        import_video(MemoryVideos(), SimpleNamespace(gallery_id="personal"), path)


def test_size_limit_before_probe(tmp_path, monkeypatch):
    path = tmp_path / "too-big.mov"
    with path.open("wb") as output:
        output.truncate(MAX_VIDEO_BYTES + 1)
    monkeypatch.setattr(video_ingest, "_tool", lambda name: pytest.fail("Must reject size before tool discovery"))
    with pytest.raises(VideoImportError, match="64 MiB"):
        import_video(MemoryVideos(), SimpleNamespace(gallery_id="personal"), path)


def test_duration_limit_and_audio_only():
    header = {"format": {"duration": "120.001"}, "streams": [{"codec_type": "video", "index": 0, "width": 320, "height": 240}]}
    with pytest.raises(VideoImportError, match="120 seconds"):
        video_ingest._video_info(header)
    with pytest.raises(VideoImportError, match="no playable video"):
        video_ingest._video_info({"format": {"duration": "4"}, "streams": [{"codec_type": "audio"}]})


def test_duplicate_returns_existing_without_ffmpeg_or_ai(tmp_path, monkeypatch):
    path = tmp_path / "original.mov"
    payload = b"already imported original bytes"
    path.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    video_id = "video-" + hashlib.sha256(f"{sha}:{VIDEO_IMPORT_VERSION}".encode()).hexdigest()[:20]
    store = MemoryVideos()
    previous = {"video_id": video_id, "gallery_id": "personal", "index_status": "indexed"}
    store.documents[("personal", video_id)] = previous
    monkeypatch.setattr(video_ingest, "_tool", lambda name: pytest.fail("Duplicate must not invoke processors"))
    assert import_video(store, SimpleNamespace(gallery_id="personal"), path) is previous


def test_failed_processing_cleans_temp_directory_and_does_not_commit(tmp_path, monkeypatch):
    path = tmp_path / "bad.mov"
    path.write_bytes(b"original")
    store = MemoryVideos()
    directories = []
    monkeypatch.setattr(video_ingest, "_tool", lambda name: name)
    monkeypatch.setattr(video_ingest, "_probe", lambda *args: {"format": {"duration": "1"}, "streams": [
        {"codec_type": "video", "index": 0, "width": 320, "height": 240}]})

    def failing(command, *, timeout, cwd, capture=False):
        directories.append(cwd)
        assert "-protocol_whitelist" in command
        assert command[command.index("-protocol_whitelist") + 1] == "file,pipe"
        assert command[command.index("-enable_drefs") + 1] == "0"
        assert str(path) not in command
        raise VideoImportError("decoder failed")

    monkeypatch.setattr(video_ingest, "_run", failing)
    with pytest.raises(VideoImportError, match="decoder failed"):
        import_video(store, SimpleNamespace(gallery_id="personal"), path)
    assert directories and all(not directory.exists() for directory in directories)
    assert not store.documents
    assert path.read_bytes() == b"original"


def test_real_color_video_import_and_resume(tmp_path, monkeypatch):
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if not Path(ffmpeg).is_file():
        pytest.skip("FFmpeg is not installed")
    path = tmp_path / "color.mp4"
    subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "color=c=blue:s=1920x1080:r=30",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "1.2",
                    "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
                   stdin=subprocess.DEVNULL, capture_output=True, check=True, timeout=30, shell=False)
    original = path.read_bytes()
    store = MemoryVideos()
    settings = SimpleNamespace(gallery_id="personal")
    document = import_video(store, settings, path)
    saved_original, playback, segments = store.assets[("personal", document["video_id"])]
    assert saved_original == original == path.read_bytes()
    assert document["source"]["attribution"] == "사용자 제공 영상"
    assert str(path) not in str(document)
    assert document["metadata"]["location"] is None
    assert document["mime_type"] == "video/mp4"
    assert document["width"] <= 1280 and document["height"] <= 720
    assert playback.find(b"moov") < playback.find(b"mdat")  # faststart
    assert len(segments) == 1 and len(segments[0]["frames"]) == 2
    assert segments[0]["end_ms"] == document["duration_ms"]
    for frame, sha in zip(segments[0]["frames"], segments[0]["frame_sha256s"], strict=True):
        assert hashlib.sha256(frame).hexdigest() == sha
        with Image.open(io.BytesIO(frame)) as image:
            assert image.format == "JPEG"
    monkeypatch.setattr(video_ingest, "_tool", lambda name: pytest.fail("Resume must skip FFmpeg"))
    assert import_video(store, settings, path) is document
