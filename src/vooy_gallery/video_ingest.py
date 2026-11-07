"""Bounded local MP4/MOV ingestion, playback conversion and deterministic sampling."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import shutil
import stat
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

VIDEO_IMPORT_VERSION = "local-video-v1-10s-2frames-h264aac-720p"
MAX_VIDEO_BYTES = 64 * 1024 * 1024
MAX_DURATION_MS = 120_000
MAX_INPUT_PIXELS = 4096 * 2304
SEGMENT_MS = 10_000
IMPORT_TIMEOUT_SECONDS = 300
INPUT_OPTIONS = ["-protocol_whitelist", "file,pipe", "-f", "mov", "-enable_drefs", "0", "-use_absolute_path", "0"]
SCALE_FILTER = "scale=w='min(1280,iw)':h='min(720,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1"


class VideoImportError(ValueError):
    """A local video failed a bounded import; no partial database record is saved."""


def _tool(name: str) -> str:
    found = shutil.which(name)
    if not found and Path(f"/opt/homebrew/bin/{name}").is_file():
        found = f"/opt/homebrew/bin/{name}"
    if not found:
        raise VideoImportError(f"{name} is required to import video.")
    return found


def _run(command: list[str], *, timeout: float, cwd: Path, capture: bool = False) -> bytes:
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                                stderr=subprocess.PIPE, cwd=cwd, timeout=timeout, check=False, shell=False)
    except subprocess.TimeoutExpired as error:
        raise VideoImportError("Video processing exceeded its time limit.") from error
    except OSError as error:
        raise VideoImportError("Unable to start the video processor.") from error
    if result.returncode:
        # Inputs are copied to generic temporary names; never surface a user's
        # absolute original path through FFmpeg diagnostics or public documents.
        detail = result.stderr.decode("utf-8", errors="replace")[-1500:]
        raise VideoImportError(f"Video processing failed: {detail}")
    return result.stdout if capture else b""


def _probe(path: Path, binary: str, deadline: float) -> dict[str, Any]:
    payload = _run([binary, "-v", "error", "-max_alloc", "268435456", *INPUT_OPTIONS,
                    "-show_entries", "format=format_name,duration:format_tags:stream=index,codec_type,codec_name,width,height,duration:stream_disposition=attached_pic",
                    "-of", "json", str(path)], timeout=_remaining(deadline, 20), cwd=path.parent, capture=True)
    if len(payload) > 1024 * 1024:
        raise VideoImportError("Video metadata is too large.")
    try:
        return json.loads(payload)
    except (ValueError, TypeError) as error:
        raise VideoImportError("Video metadata is invalid.") from error


def _remaining(deadline: float, maximum: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise VideoImportError("Video import exceeded its overall time limit.")
    return min(remaining, maximum)


def _video_info(probe: dict[str, Any]) -> tuple[dict[str, Any], int]:
    streams = probe.get("streams", [])
    videos = [stream for stream in streams if stream.get("codec_type") == "video"
              and not stream.get("disposition", {}).get("attached_pic")]
    if not videos:
        raise VideoImportError("The file has no playable video stream.")
    stream = videos[0]
    try:
        width, height, index = int(stream["width"]), int(stream["height"]), int(stream["index"])
        if min(width, height) < 2 or width * height > MAX_INPUT_PIXELS or index < 0:
            raise ValueError
    except (ValueError, TypeError, KeyError) as error:
        raise VideoImportError("Video dimensions exceed the supported 4K input bounds or are invalid.") from error
    durations = []
    for value in [probe.get("format", {}).get("duration"), *(item.get("duration") for item in streams)]:
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(duration) and duration > 0:
            durations.append(duration)
    if not durations:
        raise VideoImportError("Video duration is missing or invalid.")
    duration = max(durations)
    if duration > MAX_DURATION_MS / 1000:
        raise VideoImportError("Videos must be no longer than 120 seconds.")
    return stream, max(1, math.ceil(duration * 1000))


def embedded_metadata(tags: dict[str, Any]) -> dict[str, Any]:
    """Read an explicit capture timestamp and GPS; never infer a city.

    Generic format.creation_time may describe an encode/export operation. It is
    retained in source metadata but never substituted for a capture timestamp.
    """
    captured_at = None
    value = tags.get("com.apple.quicktime.creationdate")
    if isinstance(value, str):
        try:
            stamp = datetime.fromisoformat(value.strip())
            if stamp.tzinfo is not None and stamp.utcoffset() is not None:
                captured_at = stamp.isoformat()
        except ValueError:
            pass
    latitude = longitude = None
    raw_location = tags.get("com.apple.quicktime.location.ISO6709") or tags.get("location")
    if isinstance(raw_location, str):
        matched = re.fullmatch(r"\s*([+-]\d{2}(?:\.\d+)?)([+-]\d{3}(?:\.\d+)?)(?:[+-]\d+(?:\.\d+)?)?/?\s*", raw_location)
        if matched:
            lat, lon = map(float, matched.groups())
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                latitude, longitude = lat, lon
    return {"origin": "embedded" if captured_at else "unknown", "captured_at": captured_at,
            "location": None, "latitude": latitude, "longitude": longitude}


def segment_plan(duration_ms: int) -> list[dict[str, Any]]:
    if not 0 < duration_ms <= MAX_DURATION_MS:
        raise VideoImportError("Invalid segment duration.")
    result = []
    for start in range(0, duration_ms, SEGMENT_MS):
        end = min(start + SEGMENT_MS, duration_ms)
        midpoint = min(end - 1, start + (end - start) // 2)
        result.append({"start_ms": start, "end_ms": end, "frame_timestamps_ms": [start, midpoint]})
    return result


def _read_input(path: Path) -> tuple[Path, bytes]:
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", str(path)):
        raise VideoImportError("Only local MP4/MOV files can be imported.")
    if path.suffix.lower() not in {".mp4", ".mov"}:
        raise VideoImportError("Only .mp4 and .mov files are supported.")
    try:
        resolved = path.expanduser().resolve(strict=True)
        info = resolved.stat()
        if not stat.S_ISREG(info.st_mode):
            raise VideoImportError("The input must be a regular local file.")
        if not 0 < info.st_size <= MAX_VIDEO_BYTES:
            raise VideoImportError("Video input must be nonempty and at most 64 MiB.")
        with resolved.open("rb") as handle:
            data = handle.read(MAX_VIDEO_BYTES + 1)
    except OSError as error:
        raise VideoImportError("The local video cannot be read.") from error
    if not 0 < len(data) <= MAX_VIDEO_BYTES:
        raise VideoImportError("Video input must be nonempty and at most 64 MiB.")
    return resolved, data


def import_video(video_store: Any, settings: Any, path: Path, *, title: str | None = None,
                 source: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Atomically import original, browser playback and sampled JPEG segment frames.

    A duplicate hash/version returns the existing document before any FFmpeg
    invocation. No face recognition, transcript, audio-model or OpenAI call runs.
    """
    path, original_bytes = _read_input(Path(path))
    digest = hashlib.sha256(original_bytes).hexdigest()
    identity = hashlib.sha256(f"{digest}:{VIDEO_IMPORT_VERSION}".encode()).hexdigest()
    video_id = "video-" + identity[:20]
    gallery_id = settings.gallery_id
    existing = video_store.get_video(video_id, gallery_id)
    if existing is not None:
        return existing
    ffmpeg, ffprobe = _tool("ffmpeg"), _tool("ffprobe")
    deadline = time.monotonic() + IMPORT_TIMEOUT_SECONDS
    with tempfile.TemporaryDirectory(prefix="vooy-video-") as directory:
        workdir = Path(directory)
        original_path = workdir / ("input" + path.suffix.lower())
        playback_path = workdir / "playback.mp4"
        original_path.write_bytes(original_bytes)
        probed = _probe(original_path, ffprobe, deadline)
        stream, duration_ms = _video_info(probed)
        _run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-max_alloc", "268435456",
              "-threads", "2", "-filter_threads", "1", *INPUT_OPTIONS, "-i", str(original_path),
              "-map", f"0:{int(stream['index'])}", "-map", "0:a:0?", "-map_metadata", "-1", "-map_chapters", "-1",
              "-vf", SCALE_FILTER + ",fps=30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
              "-threads", "2", "-pix_fmt", "yuv420p", "-maxrate", "2500k", "-bufsize", "5000k",
              "-c:a", "aac", "-b:a", "128k", "-ac", "2", "-ar", "48000", "-sn", "-dn",
              "-t", f"{duration_ms / 1000:.3f}", "-fs", str(MAX_VIDEO_BYTES), "-movflags", "+faststart",
              str(playback_path)], timeout=_remaining(deadline, 180), cwd=workdir)
        if not playback_path.is_file() or not 0 < playback_path.stat().st_size <= MAX_VIDEO_BYTES:
            raise VideoImportError("Playback conversion did not produce a bounded MP4.")
        playback_probe = _probe(playback_path, ffprobe, deadline)
        playback_stream, playback_duration = _video_info(playback_probe)
        if playback_stream.get("codec_name") != "h264" or int(playback_stream["width"]) > 1280 or int(playback_stream["height"]) > 720:
            raise VideoImportError("Playback conversion did not produce 720p H.264 video.")
        if abs(playback_duration - duration_ms) > 250:
            raise VideoImportError("Playback duration differs from the source; refusing a truncated import.")
        if any(item.get("codec_name") != "aac" for item in playback_probe.get("streams", []) if item.get("codec_type") == "audio"):
            raise VideoImportError("Playback audio is not AAC.")
        segments = []
        for planned in segment_plan(duration_ms):
            segment_identity = hashlib.sha256(f"{digest}:{VIDEO_IMPORT_VERSION}:{planned['start_ms']}:{planned['end_ms']}".encode()).hexdigest()
            segment = {"segment_id": "segment-" + segment_identity[:24], **planned,
                       "frames": [], "frame_sha256s": [], "caption": None, "index_status": "pending", "identities": []}
            for index, timestamp in enumerate(planned["frame_timestamps_ms"]):
                frame_path = workdir / f"{segment['segment_id']}-{index}.jpg"
                _run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-max_alloc", "268435456",
                      "-threads", "2", "-filter_threads", "1", *INPUT_OPTIONS,
                      "-ss", f"{timestamp / 1000:.3f}", "-i", str(playback_path), "-map", "0:v:0",
                      "-frames:v", "1", "-an", "-sn", "-dn", "-q:v", "3", str(frame_path)],
                     timeout=_remaining(deadline, 20), cwd=workdir)
                if not frame_path.is_file() or not 0 < frame_path.stat().st_size <= 4 * 1024 * 1024:
                    raise VideoImportError("A requested segment frame could not be sampled.")
                frame = frame_path.read_bytes()
                with Image.open(io.BytesIO(frame)) as image:
                    if image.format != "JPEG":
                        raise VideoImportError("A sampled frame is not JPEG.")
                    image.verify()
                segment["frames"].append(frame)
                segment["frame_sha256s"].append(hashlib.sha256(frame).hexdigest())
            segments.append(segment)
        tags = probed.get("format", {}).get("tags", {})
        safe_source = {"origin": "local", "attribution": "사용자 제공 영상", **(source or {})}
        for key in ("path", "input_path", "file_path", "local_path", "absolute_path"):
            safe_source.pop(key, None)
        safe_source["embedded_metadata"] = {"format_tags": tags}
        resolved_metadata = embedded_metadata(tags)
        if metadata:
            override_origin = metadata.get("origin", "user_provided")
            if override_origin not in {"unknown", "user_provided", "synthetic", "embedded"}:
                raise VideoImportError("Unsupported metadata origin.")
            if override_origin == "embedded" and (resolved_metadata["origin"] != "embedded" or any(
                    key != "origin" and value != resolved_metadata.get(key) for key, value in metadata.items())):
                raise VideoImportError("Overridden metadata must use user_provided or synthetic origin.")
            resolved_metadata = {**resolved_metadata, **metadata, "origin": override_origin}
            if resolved_metadata.get("captured_at"):
                try:
                    stamp = datetime.fromisoformat(resolved_metadata["captured_at"])
                    if stamp.tzinfo is None or stamp.utcoffset() is None:
                        raise ValueError
                except (TypeError, ValueError) as error:
                    raise VideoImportError("Provided captured_at must be a timezone-aware ISO timestamp.") from error
        document = {"video_id": video_id, "gallery_id": gallery_id, "title": title or path.stem,
                    "sha256": digest, "duration_ms": duration_ms, "source": safe_source,
                    "metadata": resolved_metadata,
                    "mime_type": "video/quicktime" if path.suffix.lower() == ".mov" else "video/mp4",
                    "playback_mime_type": "video/mp4", "index_status": "pending", "import_version": VIDEO_IMPORT_VERSION,
                    "segment_count": len(segments), "width": int(playback_stream["width"]), "height": int(playback_stream["height"])}
        return video_store.put_video(document, original_bytes, playback_path.read_bytes(), segments)
