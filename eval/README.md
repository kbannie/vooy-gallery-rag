# 평가 데이터

이 디렉터리는 로컬 검수 데이터에 연결된 평가 질문과 수정 이력을 보관합니다. 원본 사진·개인 영상·DB와 `data/` 아래 결과 로그는 Git 저장소에 포함하지 않습니다. 새로 수집한 데이터에서는 정답 ID·메타데이터·fingerprint를 다시 검수한 뒤 평가하세요.

2026-11-07 기준 `bts-daily`의 **100장 수집·검수·설명 생성·pgvector 인덱싱과 최종 평가를 완료**했습니다. 15개 출처의 다운로드 후보 146장 중 46장을 제외했으며, 최종 사진에는 출처로 확인한 BTS 7명의 인물 태그가 모두 포함됩니다. 실제 이미지 미리보기는 [gallery.html](../data/previews/gallery.html)입니다. [data/selected/](../data/selected/)에는 열람용 원본 100장과 `manifest.json`이 있으며, 기준 저장소는 PostgreSQL입니다.

[bts_daily_queries.jsonl](bts_daily_queries.jsonl)은 완성된 40문항 평가셋입니다. 개발 10개와 최종 30개로 나뉘며, 두 집합의 정답 사진 ID는 겹치지 않습니다. 사진과 synthetic 메타데이터를 직접 확인해 정답을 지정했으며 검색 결과를 정답으로 재사용하지 않았습니다.

| 평가 | 정답 있는 질문 Hit@3 | 정답 없는 질문 | 장소·날짜·인물 필터 위반 | 결과 로그 |
|---|---|---|---|---|
| 개발 10문항 | 8/8 | 2/2 | 0건 | [daily-eval-dev.json](../data/daily-eval-dev.json) |
| 최초 최종 30문항 | 22/25 (88%) | 5/5 | 0건 | [daily-eval-test-initial.json](../data/daily-eval-test-initial.json) |
| 질문 라벨 수정 후 재평가 30문항 | 24/25 (96%) | 5/5 | 0건 | [daily-eval-test.json](../data/daily-eval-test.json) |

최초 최종 평가와 라벨 수정 후 재평가 모두 `acceptance_passed=true`입니다. 개발 평가는 최종 30문항 규격이 아니므로 그 로그의 `acceptance_passed=false`는 예상된 값입니다. **96%는 결과를 본 뒤 질문 라벨 3개를 수정한 재평가 수치이며, 독립적인 held-out 성능이 아닙니다.** 두 수치 모두 이 소규모 검수 평가셋의 Hit@3로, 다른 인물·사진·질문으로 일반화되는 성능을 입증하지 않습니다. 장소와 날짜도 실제 촬영 정보가 아닌 가상 메타데이터입니다.

PostgreSQL 통합 테스트는 테스트용 DB를 지정해 실행합니다. 각 테스트는 UUID로 구분한 별도 스키마를 만들고 그 스키마만 정리합니다. 운영 스키마를 truncate하지 않습니다. 환경변수가 없으면 DB 테스트만 skip됩니다.

```bash
docker compose exec postgres createdb -U gallery gallery_test
TEST_DATABASE_URL=postgresql://gallery:gallery@127.0.0.1:54329/gallery_test pytest -q
```

평가는 검색 결과를 정답으로 재사용하지 않습니다. 실제 사진·가상 메타데이터를 확인해 작성한 JSONL 레코드만 실행합니다. `reviewed=true`와 갤러리 fingerprint가 있어야 하며, 이미지나 가상 메타데이터가 바뀌면 다시 검수해야 합니다.

각 줄은 다음 형식입니다. 아래 사진 ID와 fingerprint는 형식 설명용 자리표시자이며 실제 정답으로 사용할 수 없습니다. 날짜 범위는 시작 이상, 종료 미만입니다.

```json
{"id":"test-01","split":"test","query":"서울에서 카페에 앉아 있는 사진","gallery_id":"bts-daily","now":"2026-11-07T13:00:00+09:00","expected_status":"ok","relevant_photo_ids":["검수한-photo-id"],"expected_filters":{"location":"서울"},"reviewed":true,"gallery_fingerprint":"gallery status에 나온 값"}
```

평가셋을 확장할 때도 실제 사진·설명·가상 메타데이터를 보고 정답을 정한 뒤 개발과 최종 평가를 분리합니다. 최종 30개는 정답 있는 질문 25개와 정답 없는 질문 5개입니다. 검색 가능한 사진 최소 100장, 정답 있는 질문의 Hit@3 ≥ 20/25, no-match 5/5, 장소·날짜·인물 조건 위반 0건을 통과 기준으로 사용합니다. 개발 질문을 최종 평가 질문으로 재사용하지 않습니다.

`Hit@3`는 최대 3개 반환 결과 중 정답이 하나라도 포함된 질문 비율입니다. `Precision@3`는 반환된 정답 수를 **3**으로 나누며, `precision_at_returned`는 실제 반환 수로 나눕니다. 예를 들어 정답 1장만 반환하면 각각 1/3과 1입니다. 관련 정답 ID를 모두 검수하지 않았다면 precision은 과소평가될 수 있습니다.

```bash
gallery evaluate eval/bts_daily_queries.jsonl --split dev
gallery evaluate eval/bts_daily_queries.jsonl --split test
```

실행 결과는 `data/evaluations/`에 저장하며, 평가셋이 없거나 미검수 상태이면 실행에 실패합니다. 현재 완료 결과를 별도로 보존한 파일은 위 표의 `data/daily-eval-*.json`입니다. 최초 질문은 [bts_daily_queries.initial.jsonl](bts_daily_queries.initial.jsonl), 수정 후 질문은 [bts_daily_queries.jsonl](bts_daily_queries.jsonl)로 구분합니다.

## 질문 라벨 재검수와 검증 범위

최초 최종 평가에서 다음 3개 질문은 지정한 정답을 반환하지 않고 `no_match`를 반환했습니다. 이후 원본 이미지 재검수에서 질문에 잘못되거나 불명확한 시각 조건이 있음을 확인했습니다.

- `test-06`: 계단을 '실내'로 단정했지만 사진에서 실내라고 확인할 수 없음.
- `test-13`: 배의 개방된 갑판·차양 주변을 '창가'라고 표현함.
- `test-22`: 배낭 끈과 허리에 묶은 옷을 '멜빵바지', 금속 난간을 '나무 난간'으로 표현함.

최초 22/25 결과를 보존하고 세 문항의 시각 조건을 수정한 뒤 동일한 모델·검색 코드로 재평가한 결과는 24/25였습니다. 수정 내역과 원본 이미지 근거는 [label_corrections.json](label_corrections.json)에 있습니다. 최초 실패 분석은 [daily-failure-analysis.json](../data/daily-failure-analysis.json)에 보존했습니다. 세 사진 모두 SQL 필터를 통과해 재정렬 후보에 들어왔으므로 최초 세 건을 벡터 검색 누락으로 분류하지 않았습니다.

수정 후에도 `test-13`의 배 위 노을·인물 실루엣 질문은 `no_match`를 반환했습니다. 남은 불일치의 추가 원인 분석과 개선은 별도 작업입니다. 결과를 본 뒤 질문을 고쳤으므로 24/25를 새로운 독립 평가 성능으로 제시하지 않습니다. 관련 사진 ID를 모두 망라하지 않았다면 precision은 과소평가될 수 있으므로 Hit@3와 함께 해석하세요.

PostgreSQL 통합 테스트를 포함한 91개 테스트와 Ruff 검사를 통과했습니다. 실제 localhost MCP Streamable HTTP에서도 기본 갤러리 검색, 후보 썸네일, 선택한 원본 이미지의 PostgreSQL SHA-256 일치, 선택 번호·대상과 모의 영수증의 일치를 검증했습니다. [MCP 보고서](../data/daily-mcp-report.json)를 참고하세요. 실제 Vooy 메시지 전송은 검증 범위에 포함되지 않으며 수행하지 않았습니다.

## 비용 기록

개발·최종 평가가 실행된 시간대가 겹치므로 각 로그의 `estimated_usd`를 합산하면 다른 실행의 호출을 중복 계산할 수 있습니다. `gallery costs`의 작업별 API 원장을 기준으로 확인하세요. 설명 응답 파싱 실패 1건은 사용량을 회수하지 못해 `unknown_usage_calls=1`로 기록하며, 비용 합계는 이 호출을 제외한 확인 가능한 소계입니다. 인덱싱 재실행에서는 완료된 99장을 재사용하고 실패한 1장을 복구했습니다.

기존 [pilot_dev.jsonl](pilot_dev.jsonl)은 Commons 사진 6장의 `bts-demo` 개발 파일럿으로 남겨 두었습니다. 해당 결과는 현재 100장 갤러리의 평가와 별개입니다.
