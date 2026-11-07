# 사진·동영상 검색 RAG · Vooy MCP

자연어로 사진과 영상 속 장면을 찾고, Vooy에서 선택한 미디어를 확인하는 프로젝트입니다.

```text
사진 / 영상 프레임 → OpenAI Vision 설명 → 텍스트 임베딩 → pgvector
검색 질문 → 날짜·장소 필터 + 의미 검색 → 재정렬 → 최대 3개 후보 → MCP 반환
```

원본 미디어는 PostgreSQL `BYTEA`, 설명·메타데이터는 `JSONB`, 임베딩은 pgvector `vector(1536)`에 저장합니다. 기본 모델은 `gpt-6-luna`와 `text-embedding-3-small`입니다.

## 설치

Python 3.11+와 Docker가 필요합니다. 동영상 처리에는 FFmpeg가 추가로 필요합니다.

```bash
git clone https://github.com/kbannie/vooy-gallery-rag.git
cd vooy-gallery-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
[ -f .env ] || cp .env.example .env
docker compose up -d postgres
```

`.env`에 `OPENAI_API_KEY`를 설정한 뒤 `gallery db-init`을 실행합니다. 나머지 설정은 [.env.example](.env.example)을 참고하세요. API 호출 비용은 `gallery costs`로 확인할 수 있습니다.

## 사진 검색

```bash
gallery collect --target 100
gallery preview --open
gallery index --limit 10     # 설명 품질·비용 확인 후 전체 처리
gallery costs
gallery index
gallery search "서울에서 서핑보드 사이에 서 있는 사진" --open
```

기본 수집 대상은 [BTS 일상 사진 출처 목록](config/daily_sources.json)입니다. 사진의 날짜·장소는 재현 가능한 가상 정보(`synthetic`)이며, 설명은 실제 보이는 내용으로 생성합니다.

## 동영상 검색

로컬 MP4/MOV를 `data/video/`에 넣습니다. 파일당 최대 120초·64 MiB이며, 10초 구간마다 프레임 2장을 분석합니다. 음성·대사 검색은 지원하지 않습니다.

```bash
brew install ffmpeg         # macOS
gallery video-import data/video
gallery video-index
gallery video-search "차창 너머 붉은 노을이 보이는 영상"
gallery video-preview
```

영상은 원본 태그에서 촬영 시각·GPS를 읽으며, 없는 값은 미상으로 둡니다. Vooy에서 영상 도구를 사용하려면 `.env`에 `VIDEO_ENABLED=true`를 설정합니다.

## Vooy MCP 연결

서버용 토큰을 만들고 실행합니다.

```bash
python scripts/setup_mcp_token.py
gallery serve
```

다른 터미널에서 외부 접속용 임시 터널을 실행합니다.

```bash
brew install cloudflared
cloudflared tunnel --url http://127.0.0.1:8765 --http-host-header 127.0.0.1:8765
```

발급된 HTTPS 주소를 `.env`의 `MCP_PUBLIC_BASE_URL`에 **`/mcp` 없이** 설정하고 서버를 재시작합니다. [Vooy 연결 화면](https://app.vooy.com/ko/connections)에 다음 값을 등록하세요.

| 항목 | 값 |
|---|---|
| MCP 서버 주소 | 발급된 HTTPS 주소 + `/mcp` |
| 인증 헤더 이름 | `Authorization` |
| 인증 토큰 | `Bearer <MCP_ACCESS_TOKEN>` |

`search_photos`, `get_photo`를 활성화합니다. 동영상 사용 시 `search_videos`, `get_video`도 활성화하세요.

채팅에서 **“서핑보드 사이에 서 있는 사진 찾아줘” → “1번 사진 보여줘”**로 확인합니다. 미디어 링크는 기본 15분간 유효합니다. 실제 채팅 첨부·전송은 Vooy가 담당하며, 이 서버는 검색 결과와 이미지·영상 링크를 반환합니다. 사용하는 동안 서버와 터널을 실행해 두어야 합니다.

## 테스트

```bash
pytest -q
ruff check src tests
```

PostgreSQL 통합 테스트는 `TEST_DATABASE_URL` 설정이 필요합니다. 평가 절차와 결과는 [평가 문서](eval/README.md)를 참고하세요.

API 키, 사진·영상 원본, DB와 실행 로그는 Git에 포함하지 않습니다. 평가 질문은 기존 로컬 데이터에 연결되어 있어 새 데이터로 평가할 때 정답 ID와 메타데이터를 다시 검수해야 합니다.
