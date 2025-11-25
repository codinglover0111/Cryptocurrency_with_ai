# 명령

아래 명령을 따라주세요.

- 한글로 git commit을 하게 되면 글자가 깨집니다!!!
- 반드시 한국어로 작성해주세요.

## 커밋 과정

### 1단계: 변경사항 확인

```bash
git status
```

- 목적: 현재 브랜치의 변경된 파일 목록 확인

```bash
git diff --stat
```

- 목적: 변경 통계 확인 (추가/삭제된 줄 수)

### 2단계: 커밋 메시지 작성

**임시 파일 생성 (`commit_msg.txt`)**

- 이유: 한글 커밋 메시지 깨짐 방지
- 방법: `-F` 옵션으로 파일에서 메시지 읽기
- 내용: 한국어로 변경사항 요약 작성

예시:

```
설정 관리 및 스토리지 시스템 리팩토링

- default_config.py: LLM 설정 구조 개선
- storage.py: DB/JSON 폴백 로직 및 에러 처리 강화
- admin.py: 설정 관리 엔드포인트 간소화
...
```

### 3단계: 파일 스테이징

```bash
git add [파일 목록]
```

- 목적: 커밋할 파일을 스테이징 영역에 추가
- **주의**: `commit_msg.txt`는 제외 (임시 파일이므로)

예시:

```bash
git add .cursor/commands/auto-commit.md AGENTS.md app/config/agents.md \
        app/config/default_config.py app/web/admin.py app/web/agents.md \
        requirements.txt static/admin.js utils/agents.md utils/storage.py
```

### 4단계: 커밋 실행

```bash
git commit -F commit_msg.txt
```

- `-F`: 파일에서 커밋 메시지 읽기
- 장점: 한글 인코딩 문제 회피, 긴 메시지 작성 용이

### 5단계: 임시 파일 삭제

```bash
# Windows PowerShell
del commit_msg.txt

# 또는 Linux/Mac
rm commit_msg.txt
```

- 목적: 작업 디렉토리 정리
- **주의**: 커밋 후 삭제 (커밋 전 삭제 시 메시지 읽기 실패)

### 6단계: 최종 확인

```bash
git log -1 --oneline
```

- 목적: 최신 커밋 확인

## 전체 프로세스 요약

```bash
# 1. 변경사항 확인
git status
git diff --stat

# 2. 커밋 메시지 파일 생성 (한국어로 작성)
# commit_msg.txt 파일을 생성하고 한국어로 메시지 작성

# 3. 변경된 파일만 스테이징 (임시 파일 제외)
git add [변경된 파일들]

# 4. 파일에서 메시지를 읽어서 커밋
git commit -F commit_msg.txt

# 5. 임시 파일 삭제
del commit_msg.txt  # Windows
# 또는
rm commit_msg.txt   # Linux/Mac
```

## 핵심 포인트

### 왜 임시 파일을 사용하나요?

- 직접 `git commit -m "한글 메시지"`를 사용하면 Windows 환경에서 한글이 깨질 수 있음
- `-F` 옵션으로 파일에서 읽으면 인코딩 문제를 피할 수 있음

### 주의사항

1. `commit_msg.txt`를 `git add`에 포함하지 않기 (임시 파일이므로)
2. 커밋 후 반드시 임시 파일 삭제
3. 파일 인코딩: UTF-8 권장
