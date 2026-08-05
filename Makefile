.PHONY: help migrate truncate rag_phase0 search rag_phase1 rag_phase2_fast rag_phase2_extra rag_phase2_save rag_phase2_save_fast rag_phase2_save_bg rag_phase3_fast rag_phase3_extra rag_phase3_save rag_phase3_save_fast rag_phase3_save_bg chat test

PY := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else command -v python; fi)

BASE_DIR ?= kb
STRATEGY ?= fixed markdown semantic
EMPRESA ?=
CLIENT_ID ?=
CHUNK_SIZE ?= 800
CHUNK_OVERLAP ?= 200
PYTEST_MARK ?= phase0
QUERY ?=
SEARCH_TYPE ?= hybrid
RERANKER ?= none
CHUNKING ?=
EVAL_OUTFILE ?= phase2
QUESTION ?=

help:
	@echo "Available commands:"
	@echo "  make migrate            - Apply sql/*.sql against the configured database"
	@echo "  make truncate           - Wipe kb_docs/kb_chunks (destructive) so you can re-ingest from scratch"
	@echo "  make rag_phase0         - Phase 0: stage kb/*.md into kb_docs, then chunk+embed into kb_chunks"
	@echo "                            default STRATEGY is all three (fixed markdown semantic); override with STRATEGY=fixed for just one"
	@echo "                            vars: BASE_DIR=kb STRATEGY=\"fixed markdown semantic\" EMPRESA=... CLIENT_ID=..."
	@echo "                                  CHUNK_SIZE=800 CHUNK_OVERLAP=200 SKIP_STAGE=1 SKIP_CHUNKS=1 TRUNCATE=1"
	@echo "  make search             - Query kb_chunks via rag/search.py's kb_search_client"
	@echo "                            vars: QUERY=... SEARCH_TYPE=hybrid|vector|text|hybrid_rrf|hybrid_union"
	@echo "                                  RERANKER=none|cohere EMPRESA=... CLIENT_ID=... CHUNKING=fixed|markdown|semantic"
	@echo "  make rag_phase1         - Phase 1: pytest -m phase1 (retrieval strategies over the materialized KB)"
	@echo "  make rag_phase2_fast    - Phase 2 (fast): RAG_FAST=1 pytest -m phase2 (~10 questions)"
	@echo "  make rag_phase2_extra   - Phase 2 (full): pytest -m phase2, all questions"
	@echo "  make rag_phase2_save    - Phase 2 runner: sweeps experiments.yaml x rag_queries.yaml, saves CSV/JSON to eval/analysis/"
	@echo "                            vars: EVAL_OUTFILE=phase2"
	@echo "  make rag_phase2_save_fast - Phase 2 runner (fast): saves CSV/JSON with a reduced question sample"
	@echo "  make rag_phase2_save_bg   - Phase 2 runner in background (nohup), logs to eval/analysis/phase2.nohup.log"
	@echo "  make rag_phase3_fast    - Phase 3 (fast): RAG_FAST=1 pytest -m phase3 (~10 questions, E2E agent + LLM-judge)"
	@echo "  make rag_phase3_extra   - Phase 3 (full): pytest -m phase3, all questions"
	@echo "  make rag_phase3_save    - Phase 3 runner: runs the agent E2E over rag_queries.yaml, saves CSV/JSON to eval/analysis/"
	@echo "  make rag_phase3_save_fast - Phase 3 runner (fast): saves CSV/JSON with a reduced question sample"
	@echo "  make rag_phase3_save_bg   - Phase 3 runner in background (nohup), logs to eval/analysis/phase3.nohup.log"
	@echo "  make chat               - Ask the ReAct RAG agent (agent/rag_agent.py) a single question"
	@echo "                            vars: QUESTION=... (requires a materialized KB under semantic chunking)"
	@echo "  make test               - Run tests/rag/ (pytest -m PYTEST_MARK, default: phase0). Truncates kb_docs/kb_chunks per test."

migrate:
	$(PY) cli.py migrate

truncate:
	$(PY) cli.py truncate

rag_phase0:
	$(PY) cli.py ingest \
		--base-dir "$(BASE_DIR)" \
		--strategy $(STRATEGY) \
		--chunk-size $(CHUNK_SIZE) \
		--chunk-overlap $(CHUNK_OVERLAP) \
		$(if $(EMPRESA),--empresa "$(EMPRESA)") \
		$(if $(CLIENT_ID),--client-id "$(CLIENT_ID)") \
		$(if $(SKIP_STAGE),--skip-stage) \
		$(if $(SKIP_CHUNKS),--skip-chunks) \
		$(if $(TRUNCATE),--truncate)

search:
	$(PY) cli.py search "$(QUERY)" \
		--search-type "$(SEARCH_TYPE)" \
		--reranker "$(RERANKER)" \
		$(if $(EMPRESA),--empresa "$(EMPRESA)") \
		$(if $(CLIENT_ID),--client-id "$(CLIENT_ID)") \
		$(if $(CHUNKING),--chunking "$(CHUNKING)")

rag_phase1:
	$(PY) -m pytest -m phase1 -v -s

rag_phase2_fast:
	RAG_FAST=1 $(PY) -m pytest -m phase2 -v -s

rag_phase2_extra:
	$(PY) -m pytest -m phase2 -v -s

rag_phase2_save:
	$(PY) cli.py eval --outfile "$(EVAL_OUTFILE)"

rag_phase2_save_fast:
	$(PY) cli.py eval --fast --outfile phase2_fast

rag_phase2_save_bg:
	nohup $(PY) cli.py eval --outfile "$(EVAL_OUTFILE)" > eval/analysis/phase2.nohup.log 2>&1 & echo $$! > eval/analysis/phase2.pid && echo "Started (PID $$(cat eval/analysis/phase2.pid))"

rag_phase3_fast:
	RAG_FAST=1 $(PY) -m pytest -m phase3 -v -s

rag_phase3_extra:
	$(PY) -m pytest -m phase3 -v -s

rag_phase3_save:
	$(PY) cli.py eval-agent --outfile phase3

rag_phase3_save_fast:
	$(PY) cli.py eval-agent --fast --outfile phase3_fast

rag_phase3_save_bg:
	nohup $(PY) cli.py eval-agent --outfile phase3 > eval/analysis/phase3.nohup.log 2>&1 & echo $$! > eval/analysis/phase3.pid && echo "Started (PID $$(cat eval/analysis/phase3.pid))"

chat:
	$(PY) cli.py chat "$(QUESTION)"

test:
	$(PY) -m pytest -m "$(PYTEST_MARK)" -v
