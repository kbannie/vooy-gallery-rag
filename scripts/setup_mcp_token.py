"""Create a separate MCP credential in .env without printing it or rotating it."""
import secrets
from pathlib import Path

from dotenv import dotenv_values, set_key


def main():
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        raise SystemExit("먼저 .env.example을 참고해 프로젝트 .env를 설정하세요.")
    values = dotenv_values(env_path)
    if not (values.get("MCP_ACCESS_TOKEN") or "").strip():
        set_key(env_path, "MCP_ACCESS_TOKEN", secrets.token_urlsafe(32))
    env_path.chmod(0o600)
    print("설정 완료: Vooy의 인증 헤더는 Authorization, 인증 토큰은 Bearer + 공백 + MCP_ACCESS_TOKEN 값입니다.")


if __name__ == "__main__":
    main()
