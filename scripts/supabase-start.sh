#!/usr/bin/env sh
set -euo pipefail

uv sync --no-dev
exec uv run main.py
