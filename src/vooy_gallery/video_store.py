"""Video originals, playback copies and frames remain in PostgreSQL BYTEA."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from psycopg import sql

from .media import validated_mime
from .store import Store, _aware_timestamp, _jsonb, _vector

_ASSETS = {
    "original": ("original_bytes", None), "playback": ("playback_bytes", "video/mp4"),
    "poster": ("poster_bytes", "image/jpeg"),
}
MAX_BYTES = 64 * 1024 * 1024
_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


class VideoStore:
    def __init__(self, base: Store):
        self.base = base
        migration = (Path(__file__).parent / "migrations/003_videos.sql").read_text()
        with base.connect() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (703015361,))
            conn.execute(migration.replace("vooy_gallery", sql.Identifier(base.schema).as_string(conn)))

    def put_video(self, document: dict, original_bytes: bytes, playback_bytes: bytes,
                  segments: list[dict]) -> dict:
        doc = dict(document)
        vid, gallery = doc["video_id"], doc["gallery_id"]
        if not _ID.fullmatch(vid) or not _ID.fullmatch(gallery):
            raise ValueError("Invalid video/gallery identifier")
        if not 0 < len(original_bytes) <= MAX_BYTES or not 0 < len(playback_bytes) <= MAX_BYTES:
            raise ValueError("Video exceeds the 64 MiB pilot limit")
        if hashlib.sha256(original_bytes).hexdigest() != doc["sha256"]:
            raise ValueError("Original video hash mismatch")
        if not 0 < doc["duration_ms"] <= 120_000 or not 1 <= len(segments) <= 24:
            raise ValueError("Invalid video duration or segment count")
        if doc["mime_type"] not in {"video/mp4", "video/quicktime"}:
            raise ValueError("Unsupported video container")
        metadata = doc.get("metadata") or {}
        if metadata.get("origin") not in {"unknown", "user_provided", "synthetic", "embedded"}:
            raise ValueError("Video metadata origin must be explicit")
        captured = _aware_timestamp(metadata.get("captured_at"))
        prepared = []
        seen = set()
        previous_end = 0
        for raw in segments:
            segment = {key: value for key, value in raw.items() if key not in {"frames", "poster_bytes"}}
            sid, start, end = segment["segment_id"], segment["start_ms"], segment["end_ms"]
            if (not _ID.fullmatch(sid) or sid in seen or start != previous_end
                    or not 0 <= start < end <= doc["duration_ms"]):
                raise ValueError("Invalid or overlapping video segments")
            seen.add(sid)
            previous_end = end
            frames = raw["frames"]
            times = segment["frame_timestamps_ms"]
            hashes = segment["frame_sha256s"]
            if not 1 <= len(frames) <= 4 or len(frames) != len(times) or len(frames) != len(hashes):
                raise ValueError("Frame evidence is incomplete")
            for frame, timestamp, digest in zip(frames, times, hashes, strict=True):
                if not start <= timestamp < end or hashlib.sha256(frame).hexdigest() != digest:
                    raise ValueError("Frame evidence mismatch")
                if validated_mime(frame, 2 * 1024 * 1024) != "image/jpeg":
                    raise ValueError("Video evidence must use JPEG frames")
            segment.update(video_id=vid, gallery_id=gallery)
            prepared.append((segment, frames))
        if previous_end != doc["duration_ms"]:
            raise ValueError("Segments must cover the whole imported clip")
        doc.update(byte_size=len(original_bytes), playback_byte_size=len(playback_bytes),
                   segment_count=len(segments), playback_mime_type="video/mp4")
        with self.base.connect() as conn:
            # The content/version ID is immutable. Resume never erases an existing index.
            inserted = conn.execute(
                "INSERT INTO videos(video_id,gallery_id,document,location,captured_at,duration_ms,"
                "original_bytes,playback_bytes,poster_bytes,original_mime) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT(video_id,gallery_id) DO NOTHING RETURNING video_id",
                (vid, gallery, _jsonb(doc), metadata.get("location"), captured, doc["duration_ms"],
                 original_bytes, playback_bytes, prepared[0][1][0], doc["mime_type"]),
            ).fetchone()
            if inserted:
                for segment, frames in prepared:
                    conn.execute(
                        "INSERT INTO video_segments(segment_id,video_id,gallery_id,document,start_ms,end_ms,"
                        "identities,frames,poster_bytes) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        (segment["segment_id"], vid, gallery, _jsonb(segment), segment["start_ms"],
                         segment["end_ms"], segment.get("identities", []), frames, frames[0]),
                    )
        return self.get_video(vid, gallery)

    def get_video(self, video_id: str, gallery_id: str) -> dict | None:
        with self.base.connect() as conn:
            row = conn.execute("SELECT document FROM videos WHERE video_id=%s AND gallery_id=%s",
                               (video_id, gallery_id)).fetchone()
        return row["document"] if row else None

    def list_videos(self, gallery_id: str) -> list[dict]:
        with self.base.connect() as conn:
            rows = conn.execute("SELECT document FROM videos WHERE gallery_id=%s ORDER BY video_id",
                                (gallery_id,)).fetchall()
        return [row["document"] for row in rows]

    @staticmethod
    def _segment(row: dict) -> dict:
        video = row["video"]
        return {**row["segment"], "metadata": video.get("metadata", {}), "source": video.get("source", {}),
                "video_title": video.get("title", ""), "video_sha256": video["sha256"]}

    def list_segments(self, gallery_id: str) -> list[dict]:
        with self.base.connect() as conn:
            rows = conn.execute(
                "SELECT s.document AS segment,v.document AS video FROM video_segments s "
                "JOIN videos v USING(video_id,gallery_id) WHERE s.gallery_id=%s ORDER BY s.video_id,s.start_ms",
                (gallery_id,),
            ).fetchall()
        return [self._segment(row) for row in rows]

    def get_segment(self, segment_id: str, gallery_id: str) -> dict | None:
        with self.base.connect() as conn:
            row = conn.execute(
                "SELECT s.document AS segment,v.document AS video FROM video_segments s "
                "JOIN videos v USING(video_id,gallery_id) WHERE s.segment_id=%s AND s.gallery_id=%s",
                (segment_id, gallery_id),
            ).fetchone()
        return self._segment(row) if row else None

    def get_segment_frames(self, segment_id: str, gallery_id: str) -> list[bytes]:
        with self.base.connect() as conn:
            row = conn.execute("SELECT frames FROM video_segments WHERE segment_id=%s AND gallery_id=%s",
                               (segment_id, gallery_id)).fetchone()
        if row is None:
            raise ValueError("Segment not found")
        return [bytes(frame) for frame in row["frames"]]

    def get_segment_poster(self, segment_id: str, gallery_id: str) -> bytes:
        with self.base.connect() as conn:
            row = conn.execute("SELECT poster_bytes FROM video_segments WHERE segment_id=%s AND gallery_id=%s",
                               (segment_id, gallery_id)).fetchone()
        if row is None:
            raise ValueError("Segment not found")
        return bytes(row["poster_bytes"])

    def video_asset_info(self, video_id: str, gallery_id: str, variant: str) -> dict | None:
        column, mime = _ASSETS[variant]
        with self.base.connect() as conn:
            row = conn.execute(sql.SQL("SELECT octet_length({}) AS size,original_mime FROM videos "
                                      "WHERE video_id=%s AND gallery_id=%s").format(sql.Identifier(column)),
                               (video_id, gallery_id)).fetchone()
        return {"size": row["size"], "mime_type": mime or row["original_mime"]} if row else None

    def get_video_asset(self, video_id: str, gallery_id: str, variant: str,
                        start: int | None = None, end: int | None = None) -> tuple[bytes, str]:
        column, mime = _ASSETS[variant]
        identifier = sql.Identifier(column)
        params = []
        if start is not None or end is not None:
            if start is None or end is None or not 0 <= start <= end:
                raise ValueError("Invalid byte range")
            body = sql.SQL("substring({} from %s for %s)").format(identifier)
            params.extend([start + 1, end - start + 1])
        else:
            body = identifier
        with self.base.connect() as conn:
            row = conn.execute(sql.SQL("SELECT {} AS data,original_mime FROM videos WHERE video_id=%s "
                                      "AND gallery_id=%s").format(body), [*params, video_id, gallery_id]).fetchone()
        if row is None:
            raise ValueError("Video not found")
        return bytes(row["data"]), mime or row["original_mime"]

    def update_segment(self, segment_id: str, gallery_id: str, caption: dict, caption_cache_key: str,
                       embedding_cache_key: str, vector, model: str) -> None:
        vector = _vector(vector)
        patch = {"caption": caption, "caption_cache_key": caption_cache_key,
                 "embedding_cache_key": embedding_cache_key, "index_status": "indexed"}
        with self.base.connect() as conn:
            row = conn.execute("UPDATE video_segments SET document=document || %s WHERE segment_id=%s "
                               "AND gallery_id=%s RETURNING video_id",
                               (_jsonb(patch), segment_id, gallery_id)).fetchone()
            if row is None:
                raise ValueError("Segment not found")
            conn.execute("INSERT INTO video_embeddings(segment_id,gallery_id,embedding,cache_key,model) "
                         "VALUES(%s,%s,%s,%s,%s) ON CONFLICT(segment_id,gallery_id) DO UPDATE SET "
                         "embedding=EXCLUDED.embedding,cache_key=EXCLUDED.cache_key,model=EXCLUDED.model",
                         (segment_id, gallery_id, vector, embedding_cache_key, model))

    @staticmethod
    def _query(gallery_id: str, filters: dict, eligible_cache_keys: dict):
        clauses, params = ["s.gallery_id=%s"], [_jsonb(eligible_cache_keys), gallery_id]
        if filters.get("location"):
            clauses.append("v.location=%s")
            params.append(filters["location"])
        for name, operator in (("captured_from", ">="), ("captured_until", "<")):
            if filters.get(name):
                clauses.append(f"v.captured_at {operator} %s")
                params.append(_aware_timestamp(filters[name]))
        if filters.get("people"):
            clauses.append("s.identities @> %s::text[]")
            params.append(filters["people"])
        query = ("FROM video_segments s JOIN videos v USING(video_id,gallery_id) "
                 "JOIN video_embeddings e USING(segment_id,gallery_id) "
                 "JOIN jsonb_each_text(%s::jsonb) AS eligible(id,key) "
                 "ON eligible.id=s.segment_id AND eligible.key=e.cache_key WHERE " + " AND ".join(clauses))
        return query, params

    def count_indexed(self, gallery_id: str, eligible_cache_keys: dict) -> int:
        query, params = self._query(gallery_id, {}, eligible_cache_keys)
        with self.base.connect() as conn:
            return conn.execute("SELECT count(*) AS count " + query, params).fetchone()["count"]

    def search_segments(self, vector, gallery_id: str, filters: dict,
                        eligible_cache_keys: dict, limit: int = 10) -> tuple[list[dict], int]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("Candidate limit must be between 1 and 100")
        query, params = self._query(gallery_id, filters, eligible_cache_keys)
        selection = "SELECT s.document AS segment,v.document AS video,"
        with self.base.connect() as conn:
            count = conn.execute("SELECT count(*) AS count " + query, params).fetchone()["count"]
            if vector is None:
                rows = conn.execute(selection + "NULL::double precision AS score " + query +
                                    " ORDER BY s.video_id,s.start_ms LIMIT %s", [*params, limit]).fetchall()
            else:
                vec = _vector(vector)
                rows = conn.execute(selection + "1-(e.embedding <=> %s) AS score " + query +
                                    " ORDER BY e.embedding <=> %s,s.segment_id LIMIT %s",
                                    [vec, *params, vec, limit]).fetchall()
        return [{"segment": self._segment(row), "score": row["score"]} for row in rows], count
