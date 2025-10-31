FROM ghcr.io/astral-sh/uv:python3.11-bookworm

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --no-dev

COPY . .

ENV TZ=Asia/Seoul \
    PYTHONUNBUFFERED=1

CMD ["uv", "run", "main.py"]
