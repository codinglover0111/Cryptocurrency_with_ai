UV ?= uv
PYTHON ?= python3.11
APP_PORT ?= 8000
FRONTEND_PORT ?= 4173

.PHONY: install backend scheduler frontend compose-up compose-down compose-logs

install:
	$(UV) pip install --python $(PYTHON) --requirement requirements.txt

backend:
	$(UV) run --python $(PYTHON) uvicorn webapp:app --host 0.0.0.0 --port $(APP_PORT)

scheduler:
	$(UV) run --python $(PYTHON) python -m main

frontend:
	NEXT_BUILD_DIR=$${NEXT_BUILD_DIR:-./web/out} $(UV) run --python $(PYTHON) python deploy/webserver.py --port $(FRONTEND_PORT)

compose-up:
	docker compose up -d --build

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f
