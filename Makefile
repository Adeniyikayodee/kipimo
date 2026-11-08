SEED ?= 7
SCENARIO ?= demo_flood
POLICY ?= lawnmower

.PHONY: test lint demo run bench train verify eval site

test:
	uv run pytest -q

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

demo:
	uv run $(if $(wildcard .env),--env-file .env,) uvicorn kipimo.server.app:app --port 8000

run:
	rm -f events.jsonl
	uv run python -m kipimo.run --scenario $(SCENARIO) --policy $(POLICY) --seed $(SEED)

bench:
	uv run --group bench python -m kipimo.bench.run


verify:
	uv run python -m kipimo.audit events.jsonl

train:
	uv run --group train python -m kipimo.perception.model --minutes 10


eval:
	uv run python -m kipimo.evals.run --agents $(or $(AGENTS),rules) --split $(or $(SPLIT),test) --trials $(or $(TRIALS),1) $(if $(MODEL),--model $(MODEL)) $(if $(FLOODS),--floods $(FLOODS))

site:
	uv run python -m kipimo.server.site --out site
