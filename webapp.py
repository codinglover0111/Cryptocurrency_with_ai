from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI

from app.api.routes import router as api_router

app = FastAPI(title="Crypto Bot API")
app.include_router(api_router, prefix="/api")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ts": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
    }
