UV ?= uv

.PHONY: sync run web compose-up compose-down

sync:
$(UV) sync

run: sync
$(UV) run main.py

web: sync
$(UV) run uvicorn webapp:app --host 0.0.0.0 --port 8000

compose-up:
docker compose up -d --build

compose-down:
docker compose down
