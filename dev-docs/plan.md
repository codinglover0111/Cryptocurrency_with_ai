# CCA 리팩토링 실행 계획

> 버전: 1.0  
> 작성일: 2026-01-02  
> 상태: 확정(리뷰 완료)

---

## Phase 0: Supabase 인프라 준비

### 작업

1. Supabase 대시보드에서 pgmq extension 활성화
2. 3개 큐 생성
   - cca_jobs_manual
   - cca_jobs_scheduled
   - cca_jobs_review
3. "Expose Queues via PostgREST" 활성화
4. pgmq_public 함수 권한: service_role만 부여
5. pgmq.q\_\* 테이블 RLS 활성화
6. cca_run_lock, cca_job_dedupe 테이블 생성

### 완료 조건

- [ ] 3개 큐가 대시보드에서 확인됨
- [ ] Python에서 supabase.schema('pgmq_public').rpc('send', {...}) 성공
- [ ] 락/디듀프 테이블 생성 및 초기 데이터 삽입 완료

### 위험/주의

- pgmq extension은 Postgres 15.6.1.143 이상에서만 사용 가능
- 운영 중 스키마 변경이므로 점검 시간 확보 권장

---

## Phase 1: Web API를 "enqueue 전용"으로 변경

### 대상 파일

- server/app/web/admin.py
  - POST /admin/run-now (라인 474)
  - POST /admin/run-symbol (라인 506)

### 작업

1. 기존 hreading.Thread 실행 로직 제거
2. cca_run_lock 상태 체크 로직 추가 (running → 409)
3. cca_job_dedupe 디듀프 체크 로직 추가 (queued → 202)
4. pgmq_public.send 호출로 교체
5. 응답 스키마 변경: { "status": "queued" | "already_queued", "run_id": "...", ... }

### 완료 조건

- [ ] run-now 호출 시 실제 실행 없이 큐에만 적재됨
- [ ] 실행 중 run-now → 409 응답
- [ ] queued 중 run-now → 202 응답

### 위험/주의

- 기존 관리자 UI 동작이 바뀌므로 프론트엔드 피드백 UI 수정 필요
- 롤백 플랜: 기존 thread 로직을 주석 처리해두고 환경변수로 분기

---

## Phase 2: Bot 스케줄러를 "enqueue 전용"으로 변경

### 대상 파일

- server/main.py
  - job() (라인 178)
  - eview_job() (라인 201)

### 작업

1. utomation_for_symbol() 직접 호출 제거
2. pgmq_public.send('cca_jobs_scheduled', {...}) 호출로 교체
3. un_loss_review() → cca_jobs_review enqueue로 교체

### 완료 조건

- [ ] 스케줄러 tick 시 실행 없이 enqueue만 발생
- [ ] 기존 로그에 "queued" 메시지 확인

### 위험/주의

- 이 시점부터 "소비자(Runner)"가 없으면 잡이 쌓이기만 함
- Phase 3과 동시 배포 권장

---

## Phase 3: Bot Runner(큐 소비자) 구현

### 대상 파일

- server/main.py (또는 별도 server/runner.py 신설)

### 작업

1. 큐 폴링 루프 구현 (우선순위: manual → scheduled → review)
2. 전역 락 획득/해제 로직 구현
3. 디듀프 상태 업데이트 로직 구현
4. 재시도/아카이브 정책 구현
5. 실패 저널 기록 로직 추가

### 완료 조건

- [ ] 큐에 메시지 적재 시 Bot이 소비하여 실행
- [ ] 동시에 2개 이상 실행 불가 (락 검증)
- [ ] 재시도 3회 초과 시 archive + 실패 저널

### 위험/주의

- Bot replica=1 강제 권장 (추가 안전장치)
- lease_expires_at 갱신(heartbeat) 구현 고려

---

## Phase 4: 레거시 UI 제거

### 대상 파일

- server/webapp.py (HTML 라우트 제거)
- server/templates/ (폴더 삭제)
- server/static/ (폴더 삭제)

### 작업

1. GET /, GET /admin, GET /overlay\*, /static 마운트 제거
2. 남은 JSON API는 server/app/web/ 라우터로 이관
3. 이관된 API에 RBAC 적용 확인
4. server/webapp.py를 "API 전용 FastAPI 앱"으로 정리

### 완료 조건

- [ ] FastAPI가 HTML을 전혀 서빙하지 않음
- [ ] 모든 UI는 Next.js에서 제공
- [ ] /admin URL 충돌 없음 (FastAPI /admin/\*는 API, Next.js /admin은 UI)

### 위험/주의

- Next.js 관리자 UI가 완전히 구현되어 있어야 함
- 점진적 제거 권장 (1개씩 제거 → 테스트 → 다음)

---

## Phase 5: 문서 정비 및 테스트

### 작업

1. AGENTS.md,
   eadme.md, server/docs/agents.md 업데이트
   - 스케줄러가 schedule 사용, 엔트리가 server/main.py 등 실제와 일치
   - Queues 기반 잡 실행 흐름 설명 추가
2. 단위 테스트 추가 (권장)
   - 락 획득/해제 로직
   - 디듀프 로직
   - 재시도/아카이브 정책
3. 통합 테스트 (권장)
   - enqueue → 소비 → 실행 → 완료 E2E

### 완료 조건

- [ ] 문서와 코드 불일치 제거
- [ ] 핵심 로직 테스트 커버리지 확보

---

## 마이그레이션 체크리스트

| 단계                      | 완료 | 담당 | 비고                     |
| ------------------------- | ---- | ---- | ------------------------ |
| Phase 0: Supabase 인프라  | ☐    |      |                          |
| Phase 1: Web API 변경     | ☐    |      | Phase 0 완료 후          |
| Phase 2: Bot enqueue 전용 | ☐    |      | Phase 1과 동시 배포 권장 |
| Phase 3: Bot Runner 구현  | ☐    |      | Phase 2와 동시 배포 필수 |
| Phase 4: 레거시 제거      | ☐    |      | Phase 3 안정화 후        |
| Phase 5: 문서/테스트      | ☐    |      | 전체 완료 후             |

---

## 롤백 계획

### Phase 1-3 롤백

- 환경변수 USE_LEGACY_EXECUTION=1 설정 시 기존 thread 실행 로직 복원
- 큐에 남은 메시지는 수동 purge 또는 archive

### Phase 4 롤백

- Git에서 삭제된 파일 복원
- /static 마운트 재활성화

---

## 일정 제안

| 주차   | 작업                        |
| ------ | --------------------------- |
| Week 1 | Phase 0 (인프라)            |
| Week 2 | Phase 1 + 2 + 3 (동시 배포) |
| Week 3 | 안정화 및 모니터링          |
| Week 4 | Phase 4 (레거시 제거)       |
| Week 5 | Phase 5 (문서/테스트)       |

---
