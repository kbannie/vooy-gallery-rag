#!/usr/bin/env python3
"""Verify local/public video MCP without search, AI calls, delivery, or secret logs.

Run only after the configured MCP server and its HTTPS tunnel are ready:
    python scripts/verify_video_mcp.py --env-file .env --report data/video-mcp-check.json

The existing server/tunnel is never started, changed, or stopped. Signed URLs and
Bearer tokens remain in memory; the report contains origins and verification
results only. PostgreSQL is queried in a read-only transaction.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

import httpx
import psycopg
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from psycopg.rows import dict_row

from vooy_gallery.settings import Settings

DEFAULT_VIDEO = "video-18eb161d674d3cf282a2"
DEFAULT_SEGMENT = "segment-04395afe1a2874afc246a266"
EXPECTED_TOOLS = {"search_photos", "get_photo", "search_videos", "get_video"}


class CheckFailure(Exception):
    """Only static, non-sensitive check names may be used as the message."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise CheckFailure(code)


def origin(value: str, *, public: bool = False) -> str:
    parsed = urlsplit(value)
    require(parsed.scheme == "https" if public else parsed.scheme in {"http", "https"}, "invalid_origin_scheme")
    require(bool(parsed.hostname) and parsed.username is None and parsed.password is None, "invalid_origin_host")
    require(parsed.path in {"", "/"} and not parsed.query and not parsed.fragment, "invalid_origin_components")
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def database_expectations(settings: Settings, video_id: str, segment_id: str) -> dict:
    with psycopg.connect(settings.database_url, connect_timeout=10, row_factory=dict_row) as connection:
        connection.execute("SET TRANSACTION READ ONLY")
        row = connection.execute(
            "SELECT v.duration_ms, octet_length(v.playback_bytes) AS playback_size, "
            "substring(v.playback_bytes from 1 for 1024) AS prefix, s.start_ms, s.end_ms "
            "FROM vooy_gallery.videos v JOIN vooy_gallery.video_segments s USING(video_id,gallery_id) "
            "WHERE v.video_id=%s AND v.gallery_id=%s AND s.segment_id=%s",
            (video_id, settings.video_gallery_id, segment_id),
        ).fetchone()
    require(row is not None, "selected_video_or_segment_missing")
    require(0 <= row["start_ms"] < row["end_ms"] <= row["duration_ms"], "invalid_database_segment_bounds")
    require(row["playback_size"] > 0, "empty_database_playback")
    row["prefix"] = bytes(row["prefix"])
    require(len(row["prefix"]) == min(1024, row["playback_size"]), "invalid_database_prefix")
    return row


def seconds(milliseconds: int) -> str:
    return f"{milliseconds // 1000}.{milliseconds % 1000:03d}".rstrip("0").rstrip(".")


async def verify_origin(
    base: str, public_base: str, settings: Settings, video_id: str, segment_id: str,
    expected: dict, report: dict,
) -> None:
    report["stage"] = "unauthenticated_mcp"
    endpoint = f"{base}/mcp"
    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as receiver:
        denied = await receiver.post(endpoint, json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
        report["unauthenticated_mcp_status"] = denied.status_code
        require(denied.status_code == 401, "mcp_did_not_require_bearer")
        report["stage"] = "mcp_initialize_and_list"
        async with (
            httpx.AsyncClient(headers={"Authorization": f"Bearer {settings.mcp_access_token}"},
                              timeout=60, follow_redirects=False) as authenticated,
            streamable_http_client(endpoint, http_client=authenticated) as streams,
            ClientSession(streams[0], streams[1]) as client,
        ):
            await client.initialize()
            listed = await client.list_tools()
            names = sorted(tool.name for tool in listed.tools)
            report["tool_names"] = names
            require(set(names) == EXPECTED_TOOLS, "unexpected_tool_catalog")
            report["stage"] = "get_selected_video"
            result = await client.call_tool("get_video", {"video_id": video_id, "segment_id": segment_id})
            require(not result.isError, "get_video_failed")
            payload = result.structuredContent
            require(isinstance(payload, dict) and payload.get("status") == "ok", "invalid_video_payload")
            require(payload.get("video_id") == video_id and payload.get("segment_id") == segment_id,
                    "selected_ids_mismatch")
            require(payload.get("gallery_id") == settings.video_gallery_id, "configured_video_gallery_mismatch")
            require(payload.get("start_ms") == expected["start_ms"] and payload.get("end_ms") == expected["end_ms"],
                    "selected_segment_bounds_mismatch")
            require(payload.get("delivery_status") == "not_sent", "unexpected_delivery_claim")
            require(payload.get("mime_type") == "video/mp4", "unexpected_playback_mime")
            require(payload.get("byte_size") == expected["playback_size"], "mcp_playback_size_mismatch")
            require(all(c.type != "image" or c.mimeType.startswith("image/") for c in result.content),
                    "non_image_base64_content")
            require(settings.mcp_access_token not in result.model_dump_json(), "bearer_token_in_tool_result")
            signed = urlsplit(payload["video_url"])
            require(origin(urlunsplit((signed.scheme, signed.netloc, "", "", "")), public=True) == public_base,
                    "unexpected_signed_media_origin")
            require(signed.path == f"/video-media/{settings.video_gallery_id}/{video_id}/playback",
                    "unexpected_signed_media_path")
            fragment = f"t={seconds(expected['start_ms'])},{seconds(expected['end_ms'])}"
            require(urlsplit(payload["playback_url"]).fragment == fragment, "playback_time_fragment_mismatch")
            require(any(c.type == "resource_link" and c.mimeType == "video/mp4" for c in result.content),
                    "missing_playback_resource_link")
            report.update(get_video_status="ok", delivery_status="not_sent", mime_type="video/mp4",
                          start_ms=expected["start_ms"], end_ms=expected["end_ms"],
                          playback_size=expected["playback_size"], segment_binding_verified=True,
                          time_fragment_verified=True, signed_url_received=True)
        # Use the same signed path/query through each transport. Signatures remain
        # bound to the configured public origin, even for this local connectivity check.
        target = urlsplit(base)
        media_url = urlunsplit(signed._replace(scheme=target.scheme, netloc=target.netloc, fragment=""))
        report["stage"] = "media_head"
        head = await receiver.head(media_url)
        report["head_status"] = head.status_code
        require(head.status_code == 200 and not head.content, "media_head_failed")
        require(head.headers.get("content-type") == "video/mp4", "media_head_mime_mismatch")
        require(int(head.headers.get("content-length", "-1")) == expected["playback_size"], "media_head_size_mismatch")
        report["stage"] = "media_range"
        end = len(expected["prefix"]) - 1
        partial = await receiver.get(media_url, headers={"Range": f"bytes=0-{end}"})
        report["range_status"] = partial.status_code
        require(partial.status_code == 206, "media_range_failed")
        require(partial.headers.get("content-range") == f"bytes 0-{end}/{expected['playback_size']}",
                "media_content_range_mismatch")
        require(partial.content == expected["prefix"], "media_range_database_bytes_mismatch")
        report.update(range_bytes=len(partial.content), range_sha256=hashlib.sha256(partial.content).hexdigest(),
                      range_matches_postgresql=True)
        report["stage"] = "corrupt_signature"
        query = parse_qs(signed.query)
        signature = query["sig"][0]
        query["sig"] = [("1" if signature.startswith("0") else "0") + signature[1:]]
        corrupt = urlunsplit(urlsplit(media_url)._replace(query=urlencode(query, doseq=True)))
        rejected = await receiver.get(corrupt)
        report["corrupt_signature_status"] = rejected.status_code
        require(rejected.status_code == 403, "corrupt_signature_not_rejected")
        report.update(status="passed", stage="complete")


async def verify_all(settings: Settings, video_id: str, segment_id: str, expected: dict, report: dict) -> None:
    public_base = origin(settings.mcp_public_base_url or "", public=True)
    host = f"[{settings.mcp_host}]" if ":" in settings.mcp_host else settings.mcp_host
    local_base = origin(f"http://{host}:{settings.mcp_port}")
    for label, base in [("local", local_base), ("public", public_base)]:
        check = {"origin_type": label, "origin": base, "status": "running"}
        report["checks"].append(check)
        try:
            await verify_origin(base, public_base, settings, video_id, segment_id, expected, check)
        except CheckFailure as exc:
            check.update(status="failed", error_code=str(exc))
        except Exception as exc:  # noqa: BLE001 -- exception text may contain signed URLs; never expose it.
            check.update(status="failed", error_type=type(exc).__name__)
    report["status"] = "passed" if all(c["status"] == "passed" for c in report["checks"]) else "failed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--report", type=Path, default=Path("data/video-mcp-check.json"))
    parser.add_argument("--video-id", default=DEFAULT_VIDEO)
    parser.add_argument("--segment-id", default=DEFAULT_SEGMENT)
    args = parser.parse_args()
    # HTTP log messages contain full request URLs. Do not let either signed media
    # credentials or nested client exception text enter script output.
    logging.disable(logging.CRITICAL)
    report = {"status": "running", "started_at": datetime.now(UTC).isoformat(),
              "video_id": args.video_id, "segment_id": args.segment_id, "checks": [],
              "search_called": False, "openai_called": False, "vooy_message_sent": False,
              "server_or_tunnel_changed": False, "signed_urls_persisted": False}
    try:
        settings = Settings.from_env(args.env_file)
        require(settings.video_enabled, "video_tools_not_enabled")
        require(bool(settings.mcp_access_token), "mcp_bearer_not_configured")
        report["gallery_id"] = settings.video_gallery_id
        expected = database_expectations(settings, args.video_id, args.segment_id)
        report["database_range_sha256"] = hashlib.sha256(expected["prefix"]).hexdigest()
        asyncio.run(verify_all(settings, args.video_id, args.segment_id, expected, report))
    except CheckFailure as exc:
        report.update(status="failed", error_code=str(exc))
    except Exception as exc:  # noqa: BLE001 -- never print SQL DSNs or HTTP credential-bearing URLs.
        report.update(status="failed", error_type=type(exc).__name__)
    report["finished_at"] = datetime.now(UTC).isoformat()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
