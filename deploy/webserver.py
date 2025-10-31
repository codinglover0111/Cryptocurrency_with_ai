"""Minimal ASGI app to serve a prebuilt frontend bundle."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

DEFAULT_PORT = 4173
DEFAULT_HOST = "0.0.0.0"
app = FastAPI(title="Crypto Bot Frontend")


@app.on_event("startup")
def _mount_frontend() -> None:
    build_dir = Path(os.getenv("NEXT_BUILD_DIR", "/srv/app/web"))
    build_dir.mkdir(parents=True, exist_ok=True)
    if not any(build_dir.iterdir()):
        raise RuntimeError(
            "Frontend bundle is empty. Provide static assets in NEXT_BUILD_DIR."
        )
    app.mount("/", StaticFiles(directory=build_dir, html=True), name="frontend")


def create_app() -> FastAPI:
    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve exported frontend assets")
    parser.add_argument("--host", default=os.getenv("HOST", DEFAULT_HOST))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", DEFAULT_PORT)),
    )
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    cli_args = _parse_args()
    uvicorn.run(
        "deploy.webserver:create_app",
        host=cli_args.host,
        port=cli_args.port,
        factory=True,
    )
