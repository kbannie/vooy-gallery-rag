import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    database_url: str = field(default="postgresql://gallery:gallery@127.0.0.1:54329/gallery", repr=False)
    api_key: str | None = field(default=None, repr=False)
    vision_model: str = "gpt-6-luna"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    reasoning_effort: str = "none"
    api_timeout: float = 60.0
    gallery_id: str = "bts-demo"
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8765
    mcp_access_token: str | None = field(default=None, repr=False)
    mcp_public_base_url: str | None = None
    mcp_image_url_ttl_seconds: int = 900
    video_enabled: bool = False
    video_gallery_id: str = "local-videos"
    vision_input_usd_per_million: float = 0.10
    vision_output_usd_per_million: float = 0.50
    embedding_input_usd_per_million: float = 0.02

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> "Settings":
        load_dotenv(env_file, override=False)
        return cls(
            data_dir=Path(os.getenv("GALLERY_DATA_DIR", "data")).expanduser().resolve(),
            database_url=os.getenv("DATABASE_URL", "postgresql://gallery:gallery@127.0.0.1:54329/gallery"),
            api_key=os.getenv("OPENAI_API_KEY", "").strip() or None,
            vision_model=os.getenv("VISION_MODEL", "gpt-6-luna"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1536")),
            reasoning_effort=os.getenv("REASONING_EFFORT", "none"),
            api_timeout=float(os.getenv("OPENAI_TIMEOUT", "60")),
            gallery_id=os.getenv("GALLERY_ID", "bts-demo"),
            mcp_host=os.getenv("MCP_HOST", "127.0.0.1"),
            mcp_port=int(os.getenv("MCP_PORT", "8765")),
            mcp_access_token=os.getenv("MCP_ACCESS_TOKEN", "").strip() or None,
            mcp_public_base_url=os.getenv("MCP_PUBLIC_BASE_URL", "").strip() or None,
            mcp_image_url_ttl_seconds=int(os.getenv("MCP_IMAGE_URL_TTL_SECONDS", "900")),
            video_enabled=os.getenv("VIDEO_ENABLED", "false").strip().lower() in {"1", "true", "yes"},
            video_gallery_id=os.getenv("VIDEO_GALLERY_ID", "local-videos"),
            vision_input_usd_per_million=float(os.getenv("VISION_INPUT_USD_PER_MILLION", "0.10")),
            vision_output_usd_per_million=float(os.getenv("VISION_OUTPUT_USD_PER_MILLION", "0.50")),
            embedding_input_usd_per_million=float(os.getenv("EMBEDDING_INPUT_USD_PER_MILLION", "0.02")),
        )
