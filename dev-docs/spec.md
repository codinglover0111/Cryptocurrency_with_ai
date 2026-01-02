# CCA 리팩토링 기술 스펙

> 버전: 1.0  
> 작성일: 2026-01-02  
> 상태: 확정(리뷰 완료)

---

1.  목표

- **레거시 제거(1.A)**: server/templates, server/static, server/webapp.py의 HTML 라우트를 완전 삭제하고 Next.js 단일 UI로 통합
- **FastAPI 직접 호출 금지(2.B)**: 브라우저는 Next.js 프록시만 통해 FastAPI 접근, FastAPI는 항상 JWT 검증
- **Supabase 단일 진실(3.A)**: 설정/프롬프트/상태 저장은 Supabase 우선, 로컬 DB는 폴백
- **동시 실행 금지(4.B)**: 어떤 경우에도 2개 이상의 트레이딩 실행이 동시에 돌지 않음

---

2.  핵심 설계 결정
    | 항목 | 결정 |
    |------|------|
    | 잡 큐 | Supabase Queues (pgmq 기반), PostgREST RPC (pgmq_public.\*)로 접근 |
    | vt_seconds (visibility timeout) | **60분** (보수적, 전체 심볼 실행 시간 고려) |
    | 실행 중 run-now 요청 | **409 Conflict** 반환 |
    | queued 중 run-now 재요청 | **202 Accepted** + "이미 대기 중" (멱등) |
    | 전역 락 | DB 테이블 cca_run_lock (lease 기반 단일 행 모델) |
    | 디듀프 | DB 테이블 cca_job_dedupe (dedupe_key 유니크) |

---

3.  인프라/DB 스키마
    3.1 Supabase Queues (3개)
    | 큐 이름 | 용도 | 우선순위 |
    |---------|------|----------|
    | cca_jobs_manual |
    un-now,
    un-symbol (관리자 수동 실행) | 1 (최우선) |
    | cca_jobs_scheduled | 스케줄러 자동 실행 | 2 |
    | cca_jobs_review | loss_review | 3 |
    3.2 전역 락 테이블: cca_run_lock
    CREATE TABLE cca_run_lock (
    lock_key TEXT PRIMARY KEY DEFAULT 'automation_global',
    status TEXT NOT NULL DEFAULT 'idle', -- 'idle' | 'running'
    holder TEXT, -- bot instance id
    run_id UUID,
    started_at TIMESTAMPTZ,
    lease_expires_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT now()
    );
    -- 초기 행 삽입
    INSERT INTO cca_run_lock (lock_key) VALUES ('automation_global');
    3.3 디듀프 테이블: cca_job_dedupe
    CREATE TABLE cca_job_dedupe (
    dedupe_key TEXT PRIMARY KEY, -- e.g. 'automation_all', 'automation_symbol:BTCUSDT'
    queue_name TEXT NOT NULL,
    msg_id BIGINT,
    status TEXT NOT NULL DEFAULT 'queued', -- 'queued' | 'running' | 'done' | 'failed'
    run_id UUID,
    requested_by TEXT, -- 'admin_api' | 'scheduler'
    requested_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
    );

---

4. 메시지 페이로드 스펙
   pgmq_public.send로 전송하는 message JSONB 구조:
   {
   job_type: automation_all | automation_symbol | loss_review,
   dedupe_key: automation_all | automation_symbol:BTCUSDT,
   symbol: BTCUSDT, // job_type=automation_symbol일 때만
   symbols: [BTCUSDT, ...], // job_type=automation_all일 때만
   requested_by: admin_api | scheduler,
   requested_at: 2026-01-02T12:00:00Z,
   run_id: uuid-v4
   }

---

5. API 스펙 변경
   5.1 POST /admin/run-now
   현재: thread로 직접 실행  
   변경: enqueue 전용 + 상태 체크
   플로우:
1. cca_run_lock.status 조회
   - running → 409 { "error": "already_running", "run_id": "..." }
1. cca_job_dedupe에서 dedupe_key='automation_all' 조회
   - 존재 & status='queued' → 202 { "status": "already_queued", "run_id": "..." }
1. 없으면:
   - pgmq_public.send('cca_jobs_manual', message, 0)
   - cca_job_dedupe INSERT
   - 202 { "status": "queued", "run_id": "...", "queue": "cca_jobs_manual" }
     5.2 POST /admin/run-symbol
     위와 동일, dedupe_key='automation_symbol:{SYMBOL}'
     5.3 스케줄러(Bot)

- 기존: 직접 automation_for_symbol() 호출
- 변경: cca_jobs_scheduled에 enqueue만

---

6. Bot Runner 상태기계
   6.1 소비 루프 (메인)
   while True:
   for queue in ['cca_jobs_manual', 'cca_jobs_scheduled', 'cca_jobs_review']:
   msg = pgmq_public.read(queue, vt_seconds=3600, n=1)
   if msg:
   process(msg)
   break
   sleep(poll_interval)
   6.2 process(msg) 플로우
1. 전역 락 획득 시도 (cca_run_lock)
   - 실패: 메시지 delete + 재-enqueue with sleep_seconds (backoff)
   - 성공: 락 획득, lease_expires_at 설정
1. cca_job_dedupe.status = 'running' 업데이트
1. 실행
   - job_type에 따라 automation_for_symbol / run_loss_review 호출
1. 결과 처리
   - 성공:
     - pgmq_public.delete(queue, msg_id)
     - cca_job_dedupe 행 삭제 또는 status='done'
     - cca_run_lock 해제 (status='idle')
   - 실패: - read_ct < MAX_RETRY: 재시도 (vt 만료 후 자동 재노출) - read_ct >= MAX_RETRY: - pgmq_public.archive(queue, msg_id) - cca_job_dedupe.status = 'failed' - 실패 저널 기록 - cca_run_lock 해제
     6.3 상수
     | 상수 | 값 | 설명 |
     |------|-----|------|
     | VT_SECONDS | 3600 (60분) | visibility timeout |
     | MAX_RETRY | 3 | 최대 재시도 횟수 |
     | POLL_INTERVAL | 5초 | 큐 폴링 주기 |
     | LEASE_DURATION | 3600초 | 락 리스 시간 |

---

7. 보안/권한
   7.1 Supabase Queues 권한

- pgmq_public.\* 함수: service_role 전용
- anon, authenticated: 권한 없음
- pgmq.q\_\* 테이블: RLS 활성화 (anon/auth 차단)
  7.2 FastAPI 라우터
- 모든 /admin/\* 라우트: require_admin 의존성 유지
- 모든 /user/\* 라우트: require_user 의존성 유지
  7.3 Next.js 프록시
- frontend/app/api/proxy/[...path]/route.ts: 세션 검증 후 Authorization: Bearer 전달
- FastAPI 직접 호출 시에도 JWT 재검증 (방화벽 우회 대비)

---

8. 레거시 제거 범위
   삭제 대상 (server/webapp.py)
   | 라인 | 라우트 | 대체 |
   |------|--------|------|
   | 204 | GET / | Next.js / |
   | 210 | GET /admin | Next.js /admin |
   | 1026 | GET /overlay | 제거 또는 Next.js 이관 |
   | 1862 | GET /overlay_positions | 제거 또는 Next.js 이관 |
   | 200 | /static 마운트 | 제거 |
   | 1874 | GET /admin/logs | Next.js Admin 또는 별도 API |
   삭제 대상 (폴더)

- server/templates/
- server/static/
  보존 대상 (API로 이관)
- /stats, /status, /api/journals\* 등은 server/app/web/ 라우터로 이관 후 RBAC 적용

---

9. 저장소 단일 진실 규칙
   | 데이터 | 우선순위 | 폴백 |
   |--------|----------|------|
   | runtime_config | Supabase runtime_config 테이블 | 로컬 DB → JSON 파일 |
   | agent_prompts | Supabase agent_prompts 테이블 | 로컬 DB → 코드 기본값 |
   | scheduler_state | Supabase scheduler_state 테이블 | 로컬 DB |
   | job_queue/lock/dedupe | Supabase Queues + 테이블 | 해당 없음 (Supabase 전용) |

---

10. 에러 처리 / 장애 시나리오
    | 시나리오 | 대응 |
    |----------|------|
    | Bot 크래시 중 락 보유 | lease_expires_at 만료 후 다른 Bot이 획득 가능 |
    | Supabase 장애 | 폴백 없이 실행 중단 (잡 큐 자체가 Supabase 의존) |
    | vt 만료 전 실행 미완료 | 메시지가 다시 visible → 다음 read에서 재시도 (read_ct 증가) |
    | 무한 재시도 | read_ct >= MAX_RETRY 시 archive + 실패 저널 |

---
