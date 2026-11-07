"""PostgreSQL stores image BYTEA, JSONB metadata and pgvector embeddings.

data_dir contains replaceable download/preview caches only. Distances and
metadata filters execute in PostgreSQL, never on a Python corpus of vectors.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import contextmanager
from datetime import datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np
import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from PIL import Image
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

DEFAULT_DATABASE_URL = "postgresql://gallery:gallery@127.0.0.1:54329/gallery"
EMBEDDING_DIMENSIONS = 1536
MIGRATION_PATH = Path(__file__).resolve().parent / "migrations" / "001_initial.sql"


class StoreError(RuntimeError):
    """Database failure safe to display without connection credentials."""


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _jsonb(value):
    return Jsonb(value, dumps=_json)


def _vector(value) -> np.ndarray:
    if isinstance(value, Vector):
        value = value.to_numpy()
    result = np.asarray(value, dtype=np.float32)
    if (result.ndim != 1 or len(result) != EMBEDDING_DIMENSIONS
            or not np.isfinite(result).all() or np.linalg.norm(result) == 0):
        raise ValueError("pgvector embeddings must be finite, nonzero vectors of exactly 1536 dimensions")
    return result


def _aware_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    timestamp = datetime.fromisoformat(value)
    if timestamp.tzinfo is None:
        raise ValueError("Timestamp filters must include a timezone")
    return timestamp


def _image_mime(data: bytes) -> str:
    with Image.open(BytesIO(data)) as image:
        mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}.get(image.format)
        if mime is None:
            raise ValueError("Unsupported gallery image format")
        image.verify()
    return mime


class Store:
    def __init__(self, data_dir: Path, database_url: str | None = None, schema: str = "vooy_gallery"):
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._database_url = database_url or os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}", schema):
            raise ValueError("Invalid PostgreSQL schema identifier")
        self.schema = schema
        self.initialize()

    @contextmanager
    def connect(self, *, register_vectors: bool = True):
        """Open a transaction with named rows; never expose a DSN in error output."""
        try:
            with psycopg.connect(self._database_url, connect_timeout=5, row_factory=dict_row) as conn:
                conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(self.schema)))
                if register_vectors:
                    register_vector(conn)
                yield conn
        except psycopg.Error:
            raise StoreError(
                "PostgreSQL operation failed. Check DATABASE_URL, database availability, "
                "and the pgvector extension. Connection credentials are not shown."
            ) from None

    def initialize(self):
        """Apply the idempotent first migration under a database transaction lock."""
        migration = MIGRATION_PATH.read_text(encoding="utf-8")
        with self.connect(register_vectors=False) as conn:
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (703015361,))
            # The migration is trusted SQL; only its fixed schema identifier is substituted.
            conn.execute(migration.replace("vooy_gallery", sql.Identifier(self.schema).as_string(conn)))
            usage_migration = (MIGRATION_PATH.parent / "002_unknown_usage.sql").read_text(encoding="utf-8")
            conn.execute(usage_migration.replace("vooy_gallery", sql.Identifier(self.schema).as_string(conn)))

    def _cache_path(self, relative: str) -> Path:
        relative_path = Path(relative)
        if relative_path.is_absolute():
            raise ValueError("Image cache paths must be relative to data_dir")
        path = (self.data_dir / relative_path).resolve()
        if not path.is_relative_to(self.data_dir) or path == self.data_dir:
            raise ValueError("Image cache path is outside gallery data directory")
        return path

    def restore_cached_assets(self, document: dict) -> dict:
        """Explicitly export missing disposable caches from authoritative BYTEA rows.

        Metadata reads, search, and get_asset do not require a writable cache.
        """
        for key, thumbnail in (("image_path", False), ("thumbnail_path", True)):
            relative = document.get(key)
            if not relative:
                continue
            path = self._cache_path(relative)
            if not path.is_file():
                data, _ = self.get_asset(document["photo_id"], document["gallery_id"], thumbnail)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
                try:
                    temporary.write_bytes(data)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
        return document

    def upsert_photo(self, photo: dict):
        document = json.loads(_json(photo))
        image_path = self._cache_path(document["image_path"])
        thumbnail_path = self._cache_path(document.get("thumbnail_path", document["image_path"]))
        document.setdefault("thumbnail_path", document["image_path"])
        metadata = document.get("metadata") or {}
        captured_at = None
        try:
            captured_at = _aware_timestamp(metadata.get("captured_at"))
        except (ValueError, TypeError):
            # Offset-less EXIF times stay in JSONB but cannot pass date filters.
            pass
        identities = document.get("identities") or []
        if not isinstance(identities, list) or not all(isinstance(name, str) for name in identities):
            raise ValueError("identities must be a list of strings")
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT gallery_id, document FROM photos WHERE photo_id=%s FOR UPDATE",
                (document["photo_id"],),
            ).fetchone()
            if existing:
                if existing["gallery_id"] != document["gallery_id"]:
                    raise ValueError("A photo cannot be moved to another gallery by upsert")
                # Caption updates must not replace authoritative bytes with stale caches.
                for name in ("image_path", "thumbnail_path", "sha256", "mime_type"):
                    if name in existing["document"]:
                        document[name] = existing["document"][name]
                conn.execute(
                    "UPDATE photos SET document=%s, location=%s, captured_at=%s, identities=%s, "
                    "updated_at=now() WHERE photo_id=%s",
                    (_jsonb(document), metadata.get("location"), captured_at, identities, document["photo_id"]),
                )
            else:
                if not image_path.is_file() or not thumbnail_path.is_file():
                    raise ValueError("First ingestion requires the downloaded image and thumbnail")
                original, thumbnail = image_path.read_bytes(), thumbnail_path.read_bytes()
                original_mime, thumbnail_mime = _image_mime(original), _image_mime(thumbnail)
                digest = hashlib.sha256(original).hexdigest()
                document["sha256"] = digest
                document["mime_type"] = original_mime
                conn.execute(
                    "INSERT INTO photos (photo_id,gallery_id,document,location,captured_at,identities,"
                    "image_bytes,thumbnail_bytes,mime_type,thumbnail_mime_type,sha256) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (document["photo_id"], document["gallery_id"], _jsonb(document), metadata.get("location"),
                     captured_at, identities, original, thumbnail, original_mime, thumbnail_mime, digest),
                )

    def get_photo(self, photo_id: str, gallery_id: str = "bts-demo") -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT document FROM photos WHERE photo_id=%s AND gallery_id=%s", (photo_id, gallery_id)
            ).fetchone()
        return row["document"] if row else None

    def list_photos(self, gallery_id: str = "bts-demo") -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT document FROM photos WHERE gallery_id=%s ORDER BY photo_id", (gallery_id,)
            ).fetchall()
        return [row["document"] for row in rows]

    def get_asset(self, photo_id: str, gallery_id: str = "bts-demo", thumbnail: bool = False) -> tuple[bytes, str]:
        # Identifiers are selected from this fixed pair, never interpolated caller input.
        query = (
            "SELECT thumbnail_bytes AS data,thumbnail_mime_type AS mime FROM photos "
            "WHERE photo_id=%s AND gallery_id=%s"
            if thumbnail else
            "SELECT image_bytes AS data,mime_type AS mime FROM photos WHERE photo_id=%s AND gallery_id=%s"
        )
        with self.connect() as conn:
            row = conn.execute(query, (photo_id, gallery_id)).fetchone()
        if row is None:
            raise ValueError("The requested photo does not exist in this gallery")
        return bytes(row["data"]), row["mime"]

    def get_cache(self, key: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM cache WHERE key=%s", (key,)).fetchone()
        return row["value"] if row else None

    def set_cache(self, key: str, value: dict):
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO cache (key,value) VALUES (%s,%s) ON CONFLICT(key) DO UPDATE "
                "SET value=excluded.value,created_at=now()", (key, _jsonb(value)),
            )

    def set_embedding(self, photo_id: str, vector, cache_key: str, model: str):
        vector = _vector(vector)
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO embeddings (photo_id,embedding,cache_key,model) VALUES (%s,%s,%s,%s) "
                "ON CONFLICT(photo_id) DO UPDATE SET embedding=excluded.embedding,"
                "cache_key=excluded.cache_key,model=excluded.model,updated_at=now()",
                (photo_id, vector, cache_key, model),
            )

    def get_embedding(self, photo_id: str, cache_key: str | None = None):
        with self.connect() as conn:
            row = conn.execute("SELECT embedding,cache_key FROM embeddings WHERE photo_id=%s", (photo_id,)).fetchone()
        if not row or (cache_key is not None and row["cache_key"] != cache_key):
            return None
        return _vector(row["embedding"]).copy()

    @staticmethod
    def _candidate_query(gallery_id: str, filters: dict, eligible_cache_keys: dict[str, str]):
        clauses = ["p.gallery_id=%s"]
        parameters: list = [_jsonb(eligible_cache_keys), gallery_id]
        if filters.get("location"):
            clauses.append("p.location=%s")
            parameters.append(filters["location"])
        for name, operator in (("captured_from", ">="), ("captured_until", "<")):
            if filters.get(name):
                clauses.append(f"p.captured_at {operator} %s")
                parameters.append(_aware_timestamp(filters[name]))
        if filters.get("people"):
            clauses.append("p.identities @> %s::text[]")
            parameters.append(filters["people"])
        query = (
            "FROM photos p JOIN embeddings e ON e.photo_id=p.photo_id "
            "JOIN jsonb_each_text(%s::jsonb) AS eligible(photo_id,cache_key) "
            "ON eligible.photo_id=p.photo_id AND eligible.cache_key=e.cache_key "
            "WHERE " + " AND ".join(clauses)
        )
        return query, parameters

    def count_indexed(self, gallery_id: str, eligible_cache_keys: dict[str, str]) -> int:
        if not eligible_cache_keys:
            return 0
        query, parameters = self._candidate_query(gallery_id, {}, eligible_cache_keys)
        with self.connect() as conn:
            row = conn.execute("SELECT count(*) AS count " + query, parameters).fetchone()
        return row["count"]

    def search_candidates(
        self, query_vector, gallery_id: str, filters: dict, eligible_cache_keys: dict[str, str], limit: int = 10,
    ) -> tuple[list[dict], int]:
        """Execute exact metadata filtering and cosine top-k entirely in PostgreSQL."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("Candidate limit must be between 1 and 100")
        vector = _vector(query_vector) if query_vector is not None else None
        if not eligible_cache_keys:
            return [], 0
        query, parameters = self._candidate_query(gallery_id, filters, eligible_cache_keys)
        with self.connect() as conn:
            count = conn.execute("SELECT count(*) AS count " + query, parameters).fetchone()["count"]
            if vector is None:
                rows = conn.execute(
                    "SELECT p.document, NULL::double precision AS score " + query
                    + " ORDER BY p.captured_at DESC NULLS LAST,p.photo_id LIMIT %s", [*parameters, limit],
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT p.document,1-(e.embedding <=> %s) AS score " + query
                    + " ORDER BY e.embedding <=> %s,p.photo_id LIMIT %s", [vector, *parameters, vector, limit],
                ).fetchall()
        return [{"photo": row["document"], "score": row["score"]} for row in rows], count

    def record_usage(self, operation: str, model: str, usage: dict, input_price: float, output_price: float = 0):
        if usage.get("usage_unavailable"):
            input_tokens = output_tokens = cost = None
        else:
            input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)))
            output_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)))
            cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO usage (operation,model,input_tokens,output_tokens,estimated_usd,details) "
                "VALUES (%s,%s,%s,%s,%s,%s)", (operation, model, input_tokens, output_tokens, cost, _jsonb(usage)),
            )

    def usage_summary(self) -> dict:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT operation,model,COUNT(*) AS calls,SUM(input_tokens) AS input_tokens,"
                "SUM(output_tokens) AS output_tokens,SUM(estimated_usd) AS estimated_usd,"
                "COUNT(*) FILTER (WHERE details->>'usage_unavailable'='true' "
                "OR input_tokens IS NULL OR output_tokens IS NULL OR estimated_usd IS NULL) AS unknown_usage_calls "
                "FROM usage GROUP BY operation,model ORDER BY operation,model"
            ).fetchall()
        groups = [{**dict(row), "calls": int(row["calls"]),
                   "input_tokens": int(row["input_tokens"]) if row["input_tokens"] is not None else None,
                   "output_tokens": int(row["output_tokens"]) if row["output_tokens"] is not None else None,
                   "estimated_usd": float(row["estimated_usd"]) if row["estimated_usd"] is not None else None,
                   "unknown_usage_calls": int(row["unknown_usage_calls"])}
                  for row in rows]
        unknown = sum(row["unknown_usage_calls"] for row in groups)
        note = "Estimate from configured standard token prices; not an invoice."
        if unknown:
            note = f"Known-usage subtotal excludes {unknown} call(s) with unavailable usage. " + note
        return {"groups": groups,
                "total_estimated_usd": sum(row["estimated_usd"] for row in groups if row["estimated_usd"] is not None),
                "unknown_usage_calls": unknown, "total_is_subtotal": bool(unknown), "cost_note": note}

    def save_search(self, result: dict) -> dict:
        result = dict(result, result_id=uuid4().hex)
        with self.connect() as conn:
            conn.execute("INSERT INTO searches (result_id,document) VALUES (%s,%s)",
                         (result["result_id"], _jsonb(result)))
        return result

    def get_search(self, result_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT document FROM searches WHERE result_id=%s", (result_id,)).fetchone()
        return row["document"] if row else None

    def save_selection(self, selection: dict) -> dict:
        selection = dict(selection, selection_id=uuid4().hex)
        with self.connect() as conn:
            conn.execute("INSERT INTO selections (selection_id,document) VALUES (%s,%s)",
                         (selection["selection_id"], _jsonb(selection)))
        return selection

    def asset_path(self, relative: str) -> Path:
        path = self._cache_path(relative)
        if not path.is_file():
            with self.connect() as conn:
                row = conn.execute(
                    "SELECT document FROM photos WHERE document->>'image_path'=%s "
                    "OR document->>'thumbnail_path'=%s LIMIT 1", (relative, relative),
                ).fetchone()
            if row:
                self.restore_cached_assets(row["document"])
        if not path.is_file():
            raise ValueError("Image cache is missing and no matching PostgreSQL asset exists")
        return path
