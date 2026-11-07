-- PostgreSQL 17 + pgvector. Photos and thumbnails reside inside PostgreSQL.
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;
CREATE SCHEMA IF NOT EXISTS vooy_gallery;
CREATE TABLE IF NOT EXISTS vooy_gallery.schema_migrations (
    version integer PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS vooy_gallery.photos (
    photo_id text PRIMARY KEY,
    gallery_id text NOT NULL,
    document jsonb NOT NULL,
    location text,
    captured_at timestamptz,
    identities text[] NOT NULL DEFAULT '{}',
    image_bytes bytea NOT NULL,
    thumbnail_bytes bytea NOT NULL,
    mime_type text NOT NULL,
    thumbnail_mime_type text NOT NULL,
    sha256 text NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK (octet_length(image_bytes) > 0),
    CHECK (octet_length(thumbnail_bytes) > 0),
    CHECK (document->>'photo_id' = photo_id),
    CHECK (document->>'gallery_id' = gallery_id)
);
CREATE INDEX IF NOT EXISTS photos_gallery ON vooy_gallery.photos(gallery_id);
CREATE INDEX IF NOT EXISTS photos_gallery_metadata ON vooy_gallery.photos(gallery_id,location,captured_at);
CREATE INDEX IF NOT EXISTS photos_identities ON vooy_gallery.photos USING gin(identities);
CREATE TABLE IF NOT EXISTS vooy_gallery.embeddings (
    photo_id text PRIMARY KEY REFERENCES vooy_gallery.photos(photo_id) ON DELETE CASCADE,
    embedding vector(1536) NOT NULL,
    cache_key text NOT NULL,
    model text NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);
-- Exact SQL cosine search is intentional for 100 photos. Approximate indexes
-- are unnecessary and can omit filter-matching candidates.
CREATE TABLE IF NOT EXISTS vooy_gallery.cache (
    key text PRIMARY KEY,
    value jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS vooy_gallery.usage (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    operation text NOT NULL,
    model text NOT NULL,
    input_tokens bigint NOT NULL,
    output_tokens bigint NOT NULL,
    estimated_usd double precision NOT NULL,
    details jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS vooy_gallery.searches (
    result_id text PRIMARY KEY,
    document jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS vooy_gallery.selections (
    selection_id text PRIMARY KEY,
    document jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
INSERT INTO vooy_gallery.schema_migrations(version) VALUES(1) ON CONFLICT(version) DO NOTHING;
