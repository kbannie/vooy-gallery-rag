# 사진·동영상 검색 RAG · Vooy MCP

사진 검색에 더해 **개인 동영상의 장면 검색**을 지원합니다. 동영상 원본·재생용 MP4·프레임은 PostgreSQL BYTEA, 구간 설명은 JSONB, 구간 벡터는 pgvector에 저장합니다. 사진은 `GALLERY_ID`, 영상은 `VIDEO_GALLERY_ID`로 구분합니다.

먼저 [설치와 실행](#설치와-실행)을 진행하세요. 이 저장소에는 코드·테스트·설정 예시·평가 질문만 포함됩니다. `.env`, 사진·영상 원본, PostgreSQL 데이터, `data/` 아래 미리보기·실행 로그는 로컬에 보관하며 Git에 포함하지 않습니다. 아래 성능 수치와 `data/` 링크는 기존 로컬 실행 기록입니다. 평가셋의 사진·영상 ID는 당시 데이터에 연결되어 있으므로 새 데이터로 평가할 때는 정답과 fingerprint를 다시 검수해야 합니다.

## 동영상 가져오기와 검색

2026-11-07 사용자 제공 MOV 3개(4.7초·5.5초·7.669초)를 실제로 저장하고 프레임 6장의 설명·임베딩을 완료했습니다. 사전 검수한 질문 8개에서 정답 있는 질문은 **6/6 Hit@3**, 정답 없는 질문은 **2/2 no_match**, 날짜 필터 위반은0건입니다. 영상 3개를 대상으로 한 기능 확인 결과이며 독립적인 대규모 검색 성능 평가는 아닙니다. 전체 자동 테스트는192개 통과했습니다.

- 실제 검색 평가: `data/video-eval-report.json`
- 공개 MCP 검색→영상 선택: `data/video-search-mcp-report.json`
- 인증·재생/Range 검증: `data/video-public-mcp-report.json`
- 설명 포함 미리보기: `data/previews/videos/index.html`

평가를 다시 실행하려면 아래 명령을 사용합니다. 검색 평가에는 OpenAI API가 사용되며 캐시된 요청은 재사용됩니다. MCP 연결 검증은 검색이나 메시지 전송을 실행하지 않습니다.

```bash
.venv/bin/python scripts/evaluate_videos.py --project-root .
.venv/bin/python scripts/verify_video_mcp.py --report data/video-public-mcp-report.json
```


로컬 MP4/MOV 파일을 입력합니다. 첫 버전은 파일당 120초·64 MiB까지 지원하며, 10초 구간마다 시작과 중간의 프레임 2장을 분석합니다. 설명은 샘플 순간의 관찰이므로 빠른 동작을 놓칠 수 있고, 음성·대사 검색은 포함하지 않습니다. 원본 음성은 재생용 MP4에 보존됩니다.

```bash
# 저장소 루트에서 실행
brew install ffmpeg
.venv/bin/python -m vooy_gallery video-import data/video
.venv/bin/python -m vooy_gallery video-index
.venv/bin/python -m vooy_gallery video-status
.venv/bin/python -m vooy_gallery video-search "바다가 보이는 영상 찾아줘"
.venv/bin/python -m vooy_gallery video-preview
```

`video-import`는 폴더의 MP4/MOV를 가져오고, 같은 내용·추출 버전은 재사용합니다. `video-index`는 설명·임베딩 캐시로 중단 후 재개할 수 있습니다. 미리보기는 `data/previews/videos/index.html`과 PostgreSQL에서 내보낸 재생용 MP4를 생성하며, 이 파일들은 다시 만들 수 있습니다.

촬영 시각은 시간대가 있는 Apple QuickTime 촬영 태그를 우선 사용합니다. 컨테이너 생성 태그는 별도 원본 메타데이터로 보존하며, 촬영 시각이 확실하지 않으면 미상으로 처리합니다. GPS는 원본 태그에서 읽지만 도시명으로 자동 추측하거나 외부 지오코딩 서비스로 보내지 않습니다. 원본 태그와 검색 메타데이터는 별도로 저장하며 영상의 EXIF/QuickTime 태그를 덮어쓰지 않습니다. 재생용 복사본은 브라우저 호환 H.264/AAC MP4로 변환하고 위치 태그를 제외합니다.

Vooy에서 영상 도구를 사용하려면 `.env`에 다음을 설정하고 서버를 재시작합니다.

```dotenv
VIDEO_ENABLED=true
VIDEO_GALLERY_ID=local-videos
```

기존 인증과 `MCP_PUBLIC_BASE_URL`을 사용합니다. Vooy 연결 화면에서 도구 목록을 새로 확인해 `search_videos`, `get_video`를 추가로 활성화하세요.

```text
search_videos 도구로 바다가 보이는 영상을 찾아줘. 후보에 번호를 붙여줘.
1번 후보의 video_id와 segment_id로 get_video를 호출해 playback_url을 보여줘.
```

검색 결과에는 구간의 시작·끝(ms), 설명, 근거, 대표 프레임이 포함됩니다. `get_video`는 재생용 MP4 링크와 원본 다운로드 링크를 반환합니다. 링크는 기본15분간 유효하고 GET/HEAD 및 HTTP byte range를 지원합니다. `playback_url`의 `#t=시작,끝`은 플레이어 시간 힌트이며, 잘라낸 파일이나 해당 구간만 허용하는 접근 권한은 아닙니다. 링크를 가진 사람은 만료 전 전체 영상에 접근할 수 있습니다. Vooy가 영상 첨부를 지원하지 않으면 링크로 열 수 있으며, 서버는 메시지 전송을 성공했다고 표시하지 않습니다.

웹 검색으로 선정한 BTS 일상·여행·카페·공원 사진을 OpenAI Vision으로 설명하고, 그 설명을 텍스트 임베딩해 자연어로 사진을 찾습니다. **이미지 원본·썸네일은 PostgreSQL `BYTEA`, 메타데이터·설명은 `JSONB`, 임베딩은 pgvector `vector(1536)`**에 저장합니다. SQL의 cosine distance로 상위 후보를 찾으며 로컬 `data/`는 다운로드·미리보기 캐시입니다.

```text
공개 웹 페이지의 사진 → 출처/중복 검사 → PostgreSQL 이미지·가상 메타데이터
                     → OpenAI Vision 설명 → text embedding → pgvector

한국어 질문 → 날짜·도시·인물 조건 추출 → SQL 필터 + cosine 검색 상위 10개
           → 설명 근거 재정렬 → 최대 3장 → 선택 → 모의 공유 / MCP 이미지 반환
```

2026-11-07 기준 `bts-daily`는 **15개 출처에서 내려받은 후보 146장 중 46장을 제외하고, 검수한 100장 모두 설명 생성·임베딩을 완료**했습니다. 출처에 근거한 인물 태그에는 BTS 7명이 모두 포함됩니다. [전체 갤러리 미리보기](data/previews/gallery.html)와 [선정 원본 100장](data/selected/)에서 확인할 수 있습니다. 최종 평가의 최초 Hit@3는 **22/25(88%)**, 시각 조건이 잘못되거나 모호한 질문 3개를 수정한 재평가는 **24/25(96%)**입니다. 두 평가 모두 정답 없는 질문 **5/5**, 필터 위반 **0건**으로 설정한 완료 기준을 통과했습니다. 이는 소규모 검수 평가셋의 결과이며, 사후 라벨 수정 재평가는 독립적인 평가가 아닙니다.

## 설치와 실행

Python 3.11 이상과 Docker Desktop이 필요합니다. 개발 검증 환경은 macOS Apple Silicon / Python 3.14입니다.

```bash
git clone https://github.com/kbannie/vooy-gallery-rag.git
cd vooy-gallery-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
docker compose up -d postgres
```

처음 실행할 때 `cp .env.example .env`로 설정 파일을 만든 뒤 아래 값을 편집합니다. 이미 `.env`가 있으면 복사하지 말고 기존 파일을 편집하세요.

```dotenv
OPENAI_API_KEY=여기에_본인의_API_키
DATABASE_URL=postgresql://gallery:gallery@127.0.0.1:54329/gallery
VISION_MODEL=gpt-6-luna
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
GALLERY_ID=bts-daily
```

Compose는 프로젝트 전용 PostgreSQL 17 + pgvector를 `127.0.0.1:54329`에 실행하고 Docker volume에 데이터를 보존합니다. 위 계정은 이 로컬 데모 전용입니다. 기존 PostgreSQL을 쓰려면 `DATABASE_URL`을 바꾸고 해당 DB에서 pgvector 설치·확장이 가능해야 합니다. 데이터는 `vooy_gallery` 스키마에 저장됩니다.

```bash
gallery db-init
gallery collect --target 100
gallery preview --open
gallery index --limit 10
gallery costs
gallery index --workers 4
gallery status
```

새 갤러리는 처음 10장 결과와 비용을 확인한 뒤 전체를 인덱싱합니다. `--workers`는 동시 처리 사진 수이며 기본값 4, 허용 범위는 1~8입니다. `index` 재실행은 사진 해시·모델·프롬프트 버전·임베딩 내용을 확인해 완료 작업을 재사용합니다. 이번 실행에서는 설명 응답 파싱 실패 1건을 재실행해 복구했고, 완료된 99장은 재사용했습니다. API 키가 없으면 실제 설명이나 임베딩을 만들어낸 것처럼 대체하지 않습니다.

기본 `collect`는 웹 검색으로 미리 선정한 `config/daily_sources.json`의 공개 기사·갤러리에서 이미지 URL을 추출합니다. Google 검색 결과의 작은 미리보기를 원본처럼 저장하지 않습니다. 다른 출처 목록은 `gallery collect --sources 다른_목록.json --target 100`으로 지정할 수 있습니다. Wikimedia Commons 수집기는 선택 기능으로 남아 있습니다.

```bash
GALLERY_ID=bts-commons gallery collect --source commons --target 100
```

검수에서 그림·상품·주제에 맞지 않는 사진으로 제외된 항목이 있으면 `collect --target 100`으로 보충한 후 `index`를 다시 실행합니다. 수집 개수와 실제 검색 가능한 사진 수는 구별합니다. 네트워크 제한이나 출처 부족으로 목표에 못 미치면 `status=incomplete`와 실제 수량·오류를 반환합니다. 현재 갤러리는 무대·행사 사진, 화면 캡처, 콜라주, 인물 없는 장면, 유사 사진 등을 검수해 최종 100장을 구성했습니다.

## 검색·선택

```bash
gallery search "서울에서 검은 옷을 입고 카페에 앉아 있는 사진" --open
gallery search "서핑보드가 양옆에 세워진 통로에서 팔을 벌린 사진" --open
gallery search "지난달 부산에서 찍은 사진" --now '2026-11-07T13:00:00+09:00' --open
gallery select 검색결과_ID --number 2 --to "친구 단톡방"
```

검색 결과 JSON에 `result_id`, `status`, `applied_filters`, 후보 정보와 HTML 미리보기 경로가 나옵니다. HTML은 이미지가 포함된 단일 파일이라 별도 웹 서버가 필요 없습니다. 선택 번호는 지정한 검색 결과에 고정되어, 다른 검색을 실행해도 이전 결과의 사진이 바뀌지 않습니다.

- `ok`: 조건을 뒷받침하는 후보가 있습니다. 최대 3장이며 억지로 채우지 않습니다.
- `no_match`: 현재 인덱스에서 조건에 맞는 사진을 찾지 못했습니다.
- `needs_clarification`: “그때 거기”처럼 추가 정보가 필요합니다.
- 인덱스 미생성·API 오류는 `no_match`로 숨기지 않고 오류로 표시합니다.

`select`는 항상 `status=simulated`, `sent=false`입니다. 결과를 PostgreSQL과 `data/shares/`에 기록하며 **Vooy로 실제 메시지를 보내지 않습니다.**

## 데이터와 검색 방식

| 데이터 | 저장·처리 |
|---|---|
| 원본·썸네일 | PostgreSQL `BYTEA`; 로컬 파일을 지워도 DB에서 읽을 수 있음 |
| 출처 | 공개 페이지 URL·실제 이미지 URL·명시된 저작자/계정·라이선스·원본 설명 |
| 가상 메타데이터 | 고정 seed=42, 2026-08-01~2026-11-07, Asia/Seoul, 서울/부산/제주/인천 |
| 시각적 설명 | 실제 보이는 옷·행동·배경·인원; 얼굴로 실명 추측 안 함 |
| 인물 태그 | 출처의 명시적인 인물 근거; 얼굴에서 이름을 추측하지 않음 |
| 임베딩 | 한국어 설명·시각 태그·출처 인물 태그; 출처의 실제 날짜/도시는 제외 |
| 검색 | 장소·시각·인물 SQL 필터 → pgvector `<=>` cosine distance → 설명 근거 재정렬 |

`data/selected/`에는 최종 선정한 원본 100장과 `manifest.json`을 사용자가 둘러볼 수 있도록 내보냈습니다. 이 폴더는 열람용 사본이며 PostgreSQL이 이미지·메타데이터의 기준 저장소입니다.

가상 시간·장소는 `metadata.origin=synthetic`이며 실제 BTS 일정이 아닙니다. 원본 EXIF는 수정하지 않습니다. `metadata.py`의 EXIF provider는 실제 시각·GPS를 읽는 교체 지점입니다. EXIF 누락은 미상이며, GPS를 도시 이름으로 바꾸는 geocoding은 현재 포함하지 않습니다.

웹 사진은 최소 짧은 변 350px, Commons 사진은 480px의 JPEG/PNG/WebP를 대상으로 SHA-256과 pHash 중복을 검사합니다. 웹 수집은 요청 간격·크기 제한·호스트별 오류 기록을 적용하고, 접근 불가나 rate limit을 수집 성공으로 처리하지 않습니다. 자동 필터만으로 무대 사진·콜라주·유사 crop 등을 완벽히 구별할 수 없어 최종 미리보기 검수가 필요합니다.

웹 출처는 라이선스가 명시되지 않았으면 `license_name="Not specified"`로 보존합니다. 공식 SNS 사진을 재게시한 기사도 Creative Commons 사진으로 간주하지 않습니다. 출처·계정·원문 링크는 PostgreSQL과 로컬 미리보기에 보존합니다. Vooy 채팅용 MCP 응답에서는 출처·저작자·라이선스 필드와 자동 안내 문구를 제외하고 사진/영상과 요청한 검색 정보만 제공합니다. 선택형 Commons 수집기는 파일별 저작자와 실제 라이선스를 별도로 기록합니다.

100장 규모에서는 정확한 pgvector 검색을 사용합니다. 별도 외부 벡터 DB나 GPU는 필요하지 않습니다. `EMBEDDING_DIMENSIONS`는 DB 스키마에 맞춰 1536으로 고정합니다.

## Vooy MCP 연결 준비

```bash
gallery serve
```

로컬 endpoint: `http://127.0.0.1:8765/mcp` (Streamable HTTP). stdio 클라이언트는 `gallery serve --transport stdio`를 사용합니다.

### Vooy 연결 화면에 등록하기

사진 서버 전용 `MCP_ACCESS_TOKEN`이 설정되어 있으면 HTTP MCP는 Bearer 인증을 요구합니다. `.venv/bin/python scripts/setup_mcp_token.py`로 생성할 수 있으며, 기존 토큰은 유지합니다. OpenAI API 키는 Vooy에 입력하지 않습니다.

터미널 A에서 PostgreSQL과 사진 서버를 실행합니다.

```bash
# 저장소 루트에서 실행
docker compose up -d postgres
.venv/bin/gallery serve
```

터미널 B에서 테스트용 외부 주소를 만듭니다.

```bash
brew install cloudflared
cloudflared tunnel --url http://127.0.0.1:8765 --http-host-header 127.0.0.1:8765
```

출력된 `https://...trycloudflare.com` 주소 뒤에 `/mcp`를 붙여 [Vooy 서비스 연결](https://app.vooy.com/ko/connections)의 **MCP 서버 추가 → MCP 서버 주소**에 입력합니다. **수동 토큰 사용**을 선택하고, **인증 헤더 이름**에는 `Authorization`, **인증 토큰**에는 `Bearer ` 뒤에 `.env`의 `MCP_ACCESS_TOKEN` 값을 붙여 입력합니다. `.env`에 값을 둘러싼 따옴표가 있으면 따옴표는 제외합니다. 서버 신뢰 체크 후 **연결 확인**을 누르고, `search_photos`, `get_photo` 두 도구가 검색되고 활성화되는지 확인한 뒤 저장합니다. 기존 `ping` 연결과 별도 항목으로 등록하면 구분하기 쉽습니다.

Host 헤더 옵션은 터널이 로컬 서버에 전달하는 Host를 맞추며, MCP의 Host·Origin 검증을 유지합니다. 서버는 JSON 응답의 stateless Streamable HTTP를 사용합니다. 별도 OAuth 발급 서버를 구현한 것은 아니므로 Vooy의 수동 토큰 방식을 사용합니다.

Vooy 채팅에서 `BTS 사진 검색 도구로 서핑보드 사이에서 팔을 벌린 사진을 찾아줘`를 입력하고, 후보가 나오면 `1번 사진 보여줘`로 원본 반환을 확인합니다. 이미지가 Vooy에서 표시되는 방식과 다른 단톡방·DM으로 전송되는 동작은 실제 연결 후 별도 확인이 필요합니다.

### 사진은 찾았지만 Vooy가 첨부하지 못하는 경우

MCP의 `ImageContent`는 이미지 데이터를 도구 결과에 담습니다. Vooy가 이 결과를 채팅 첨부로 변환하는 기능은 별도입니다. 이미지 URL을 요구하는 클라이언트에는 PostgreSQL 원본을 직접 내려받는 임시 URL을 함께 제공합니다. `.env`에 현재 터널의 **origin만** 설정하고 사진 서버를 재시작하세요. 터널 프로세스는 그대로 유지합니다.

```dotenv
MCP_PUBLIC_BASE_URL=https://your-current-tunnel.trycloudflare.com
MCP_IMAGE_URL_TTL_SECONDS=900
```

`get_photo`는 기존 이미지 블록에 `image_url`, `image_url_expires_at`, 파일명, 바이트 수, SHA-256, MCP `resource_link`를 추가합니다. 검색 후보에는 `preview_url`이 추가됩니다. `/media/...` 링크는 사진·갤러리·원본/썸네일·만료 시각에 묶인 서명을 검증하고 PostgreSQL BYTEA만 읽습니다. MCP Bearer 토큰은 URL에 들어가지 않습니다. 유효한 링크를 가진 사람은 15분 동안 해당 이미지 하나를 열 수 있으므로 영구 공개 URL로 사용하지 마세요. 서명이 없거나 변조되거나 만료되면403이며, `/mcp`에는 기존 Bearer 인증이 계속 필요합니다. 서버는 서명 URL이 남지 않도록 HTTP access log를 기록하지 않습니다.

Vooy에 다음과 같이 요청해서 다시 확인합니다.

```text
방금 선택한 사진의 get_photo를 다시 호출해줘.
반환된 image_url을 사용해 사진으로 보여줘.
첨부할 수 없으면 기사 주소 대신 image_url을 클릭 가능한 링크로 보여줘.
```

링크가 만료되면 `get_photo`를 다시 호출하고, 터널 주소가 바뀌면 `MCP_PUBLIC_BASE_URL`과 Vooy 연결 URL을 갱신합니다. `image_url`은 선택한 사진 파일입니다. `delivery_status=not_sent`는 이 서버가 채팅 전송까지 수행하지 않았다는 뜻입니다. Vooy가 첨부를 거절하면 사진 링크 열기까지 이용할 수 있으며, 실제 첨부/DM 전송 성공으로 표시하지 않습니다.

Quick Tunnel은 테스트용 임시 주소이며 터널을 다시 만들면 주소가 바뀝니다. 사용하는 동안 Mac, 사진 서버와 터널을 켜 두어야 합니다. [Cloudflare Quick Tunnels 안내](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/), [HTTP Host 헤더 설정](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/configure-tunnels/origin-parameters/#httphostheader).

| 도구 | 입력 | 반환 |
|---|---|---|
| `search_photos` | `query`, `gallery_id` 생략 시 설정된 갤러리, `limit=3` | 상태·검색 결과 ID·사진 ID·설명·가상 메타데이터·썸네일 이미지 블록·설정 시 임시 `preview_url` |
| `get_photo` | `photo_id`, `gallery_id` 생략 시 설정된 갤러리 | 원본 이미지 블록·MIME·설정 시 임시 `image_url`/resource link·`delivery_status=not_sent` |

Vooy가 후보를 보여주고 사용자가 선택하면 `get_photo`를 호출하는 구조입니다. 실제 단톡방/DM 첨부·전송은 Vooy의 등록·인증·이미지 렌더링·대상 지정 명세를 확인한 뒤 연결합니다. 원격 Vooy 서비스는 사용자의 `127.0.0.1`에 직접 접속할 수 없으므로 실제 연결 단계에는 인증된 접근 경로가 필요합니다. 현재 서버는 로컬 loopback만 허용합니다.

공식 MCP Python SDK의 `ClientSession`으로 실제 localhost Streamable HTTP 연결을 검증했습니다. `gallery_id`를 생략한 검색이 `bts-daily` 후보와 썸네일을 반환했고, `get_photo`의 JPEG 211,905바이트는 PostgreSQL 원본과 SHA-256이 일치했습니다. 선택한 1번 사진과 `테스트 단톡방` 대상이 모의 영수증에 일치함도 확인했습니다. [MCP 검증 보고서](data/daily-mcp-report.json)에 기록되어 있으며 임시 테스트 서버는 종료했습니다.

## 검증과 비용

```bash
pytest -q
ruff check src tests
gallery costs
```

동영상 기능까지 포함한 전체 자동 테스트 **192개와 Ruff 검사**를 통과했습니다. DB 테스트를 실행하려면 별도 `TEST_DATABASE_URL`이 필요하며, 설정하지 않으면 해당 테스트가 skip됩니다. 설정과 평가 절차는 [eval/README.md](eval/README.md)를 참고하세요. `requirements.lock`은 구현에 사용한 환경의 패키지 버전 스냅샷입니다.

| 평가 | 정답 있는 질문 Hit@3 | 정답 없는 질문 | 필터 위반 |
|---|---|---|---|
| 개발 10문항 | 8/8 | 2/2 | 0건 |
| 최초 최종 30문항 | 22/25 (88%) | 5/5 | 0건 |
| 질문 라벨 수정 후 재평가 30문항 | 24/25 (96%) | 5/5 | 0건 |

[평가셋 40문항](eval/bts_daily_queries.jsonl)은 실제 이미지와 가상 메타데이터를 보고 작성했으며, 개발·최종 평가의 정답 사진 ID는 겹치지 않습니다. [개발 로그](data/daily-eval-dev.json), [최초 최종 평가 로그](data/daily-eval-test-initial.json), [수정 후 재평가 로그](data/daily-eval-test.json)를 각각 보존했습니다. 두 최종 평가의 `acceptance_passed=true`입니다. 최초 `no_match` 3개 문항을 원본 이미지로 재검수해 시각 조건 오류·모호성을 [수정 기록](eval/label_corrections.json)에 남겼으며, 모델과 검색 코드는 변경하지 않았습니다. 재평가에서도 배 위 노을과 인물 실루엣을 찾는 `test-13`은 `no_match`를 반환했습니다. [평가 문서](eval/README.md)에 범위와 한계를 설명했습니다. 96%를 새로운 독립 평가 성능이나 일반 사진 검색 성능으로 해석하지 않습니다.

`gallery costs`는 작업별 API 사용 토큰과 설정된 단가로 계산한 추정 USD를 제공합니다. 이미지 입력도 입력 토큰 사용량에 포함되며 청구서와는 다를 수 있습니다. 기본 단가는 Luna 입력/출력 100만 토큰당 $0.10/$0.50, 임베딩 $0.02입니다. 모델을 바꾸면 `.env`의 단가도 함께 변경하세요. 이번 파싱 실패 1건은 사용량을 회수하지 못해 `unknown_usage_calls=1`로 기록되며, 합계는 **확인된 사용량만의 소계**입니다. 개발·최종 평가 로그의 비용 측정 시간대가 겹치므로 두 `estimated_usd`를 더하지 말고, 중복 없는 작업별 원장인 `gallery costs`를 확인하세요.

주요 파일: `web_collector.py` 기본 웹 수집, `collector.py` 선택형 Commons 수집, `config/daily_sources.json` 검색으로 선정한 출처, `metadata.py` 가상/EXIF 메타데이터, `ai.py` OpenAI 호출, `retrieval.py` 인덱싱·검색, `store.py` PostgreSQL·pgvector, `mcp_server.py` MCP, `evaluation.py` 정답 기반 평가.

공식 참고: [OpenAI Vision](https://developers.openai.com/api/docs/guides/images-vision), [Embeddings](https://developers.openai.com/api/docs/guides/embeddings), [pgvector](https://github.com/pgvector/pgvector), [Commons 재사용](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia), [Vooy](https://vooy.com/en).
