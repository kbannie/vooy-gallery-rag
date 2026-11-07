"""Keep demo metadata separate from a photograph's original EXIF metadata."""

from __future__ import annotations

import hashlib
import io
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from PIL import Image

SEOUL = timezone(timedelta(hours=9))


def synthetic_metadata(photo_id: str, seed: int = 42) -> dict[str, Any]:
    """Assign a reproducible, explicitly fictional event in the demo date range."""
    number = int.from_bytes(hashlib.sha256(f"{seed}:{photo_id}".encode()).digest()[:8], "big")
    event_index = number % 16
    location = ("서울", "부산", "제주", "인천")[event_index % 4]
    event_day = date(2026, 8, 1) + timedelta(days=(event_index * 98) // 15)
    captured = datetime.combine(event_day, time(14 + event_index % 7), SEOUL)
    captured += timedelta(minutes=(number // 16) % 60)
    return {
        "origin": "synthetic",
        "captured_at": captured.isoformat(),
        "location": location,
        "event": f"가상 {location} 모임 {event_index + 1:02d}",
        "latitude": None,
        "longitude": None,
    }


def _number(value: Any) -> float:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return float(value[0]) / float(value[1])
    return float(value)


def _coordinate(value: Any, reference: Any) -> float | None:
    try:
        degrees, minutes, seconds = (_number(v) for v in value)
        coordinate = degrees + minutes / 60 + seconds / 3600
        if isinstance(reference, bytes):
            reference = reference.decode("ascii", errors="replace")
        return -coordinate if reference in ("S", "W") else coordinate
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def extract_exif(image: bytes | str | Path) -> dict[str, Any]:
    """Read source values without rewriting files or inventing a timezone/location."""
    source = io.BytesIO(image) if isinstance(image, bytes) else image
    with Image.open(source) as opened:
        exif = opened.getexif()
        try:
            detail = exif.get_ifd(34665)
        except (KeyError, TypeError, ValueError):
            detail = {}
        try:
            gps = exif.get_ifd(34853)
        except (KeyError, TypeError, ValueError):
            gps = {}
        return {
            "datetime_original": detail.get(36867, exif.get(36867)),
            "offset_time_original": detail.get(36881, exif.get(36881)),
            "latitude": _coordinate(gps.get(2), gps.get(1)),
            "longitude": _coordinate(gps.get(4), gps.get(3)),
        }


def exif_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    """Future real-photo provider; an EXIF timestamp without an offset stays naive."""
    captured = None
    original = raw.get("datetime_original")
    if isinstance(original, str):
        try:
            parsed = datetime.strptime(original.strip("\x00 "), "%Y:%m:%d %H:%M:%S")
            offset = raw.get("offset_time_original")
            if isinstance(offset, str) and re.fullmatch(r"[+-]\d{2}:\d{2}", offset.strip()):
                sign = -1 if offset.strip()[0] == "-" else 1
                hours, minutes = (int(v) for v in offset.strip()[1:].split(":"))
                if hours <= 23 and minutes <= 59:
                    parsed = parsed.replace(tzinfo=timezone(sign * timedelta(hours=hours, minutes=minutes)))
            captured = parsed.isoformat()
        except ValueError:
            pass
    return {
        "origin": "exif",
        "captured_at": captured,
        "location": None,  # GPS-to-place lookup is a separate concern.
        "event": None,
        "latitude": raw.get("latitude"),
        "longitude": raw.get("longitude"),
    }
