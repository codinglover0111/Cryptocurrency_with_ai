FROM python:3.10-slim

WORKDIR /app

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# 프로젝트 파일 복사
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 전체 프로젝트 복사 (bicoin_auto 폴더 경로 수정)
COPY . .

# 환경 설정
ENV TZ=Asia/Seoul
ENV PYTHONUNBUFFERED=1

# 실행 명령
CMD ["python", "main.py"]