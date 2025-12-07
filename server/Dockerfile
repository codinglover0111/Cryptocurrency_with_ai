FROM ghcr.io/astral-sh/uv:python3.12-alpine

WORKDIR /app

# 필수 빌드 의존성 설치
RUN apk add --no-cache \
    pkgconf \
    python3-dev \
    mariadb-dev \
    build-base \
    curl \
    tzdata

# uv를 시스템 파이썬에 사용
ENV UV_SYSTEM_PYTHON=1 \
    TZ=Asia/Seoul \
    PYTHONUNBUFFERED=1

# 의존성 설치
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# 애플리케이션 복사
COPY . .

CMD ["uv", "run", "uvicorn", "webapp:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
