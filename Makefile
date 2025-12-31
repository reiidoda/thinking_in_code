.PHONY: bootstrap api worker test lint fmt

bootstrap:
	python -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -e libs/contracts -e libs/podcast -e services/api -e services/worker

api:
	. .venv/bin/activate && uvicorn podcastify_api.main:app --reload --port 8000

worker:
	. .venv/bin/activate && python -m podcastify_worker.main

test:
	. .venv/bin/activate && pytest -q

lint:
	. .venv/bin/activate && ruff check .
	. .venv/bin/activate && mypy .

fmt:
	. .venv/bin/activate && ruff format .
