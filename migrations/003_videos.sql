CREATE TABLE IF NOT EXISTS vooy_gallery.videos (
    video_id text NOT NULL,
    gallery_id text NOT NULL,
    document jsonb NOT NULL,
    location text,
    captured_at timestamptz,
    duration_ms integer NOT NULL CHECK (duration_ms > 0 AND duration_ms <= 120000),
    original_bytes bytea NOT NULL,
    playback_bytes bytea NOT NULL,
    poster_bytes bytea NOT NULL,
    original_mime text NOT NULL,
    PRIMARY KEY(video_id, gallery_id)
);
CREATE INDEX IF NOT EXISTS videos_filters ON vooy_gallery.videos(gallery_id, location, captured_at);
CREATE TABLE IF NOT EXISTS vooy_gallery.video_segments (
    segment_id text NOT NULL,
    video_id text NOT NULL,
    gallery_id text NOT NULL,
    document jsonb NOT NULL,
    start_ms integer NOT NULL CHECK(start_ms >= 0),
    end_ms integer NOT NULL CHECK(end_ms > start_ms),
    identities text[] NOT NULL DEFAULT '{}',
    frames bytea[] NOT NULL,
    poster_bytes bytea NOT NULL,
    PRIMARY KEY(segment_id, gallery_id),
    FOREIGN KEY(video_id, gallery_id) REFERENCES vooy_gallery.videos(video_id, gallery_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS vooy_gallery.video_embeddings (
    segment_id text NOT NULL,
    gallery_id text NOT NULL,
    embedding vector(1536) NOT NULL,
    cache_key text NOT NULL,
    model text NOT NULL,
    PRIMARY KEY(segment_id, gallery_id),
    FOREIGN KEY(segment_id, gallery_id) REFERENCES vooy_gallery.video_segments(segment_id, gallery_id) ON DELETE CASCADE
);
INSERT INTO vooy_gallery.schema_migrations(version) VALUES(3) ON CONFLICT(version) DO NOTHING;
