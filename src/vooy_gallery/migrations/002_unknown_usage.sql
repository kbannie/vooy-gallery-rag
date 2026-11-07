-- SDK schema-parse failures can hide usage. NULL is unknown, not zero tokens.
ALTER TABLE vooy_gallery.usage
    ALTER COLUMN input_tokens DROP NOT NULL,
    ALTER COLUMN output_tokens DROP NOT NULL,
    ALTER COLUMN estimated_usd DROP NOT NULL;
INSERT INTO vooy_gallery.schema_migrations(version) VALUES(2) ON CONFLICT(version) DO NOTHING;
