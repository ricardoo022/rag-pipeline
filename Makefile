.PHONY: help migrate truncate ingest search eval chat test

PY := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else command -v python; fi)

BASE_DIR ?= kb
STRATEGY ?= fixed
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
	@echo "  make migrate  - Apply sql/*.sql against the configured database"
	@echo "  make truncate - Wipe kb_docs/kb_chunks (destructive) so you can re-ingest from scratch"
	@echo "  make ingest   - Stage kb/*.md into kb_docs, then chunk+embed into kb_chunks"
	@echo "                  vars: BASE_DIR=kb STRATEGY=fixed|markdown|semantic EMPRESA=... CLIENT_ID=..."
	@echo "                        CHUNK_SIZE=800 CHUNK_OVERLAP=200 SKIP_STAGE=1 SKIP_CHUNKS=1 TRUNCATE=1"
	@echo "  make search   - Query kb_chunks via rag/search.py's kb_search_client"
	@echo "                  vars: QUERY=... SEARCH_TYPE=hybrid|vector|text|hybrid_rrf|hybrid_union"
	@echo "                        RERANKER=none|cohere EMPRESA=... CLIENT_ID=... CHUNKING=fixed|markdown|semantic"
	@echo "  make eval     - Sweep tests/rag/experiments.yaml x rag_queries.yaml, report to eval/analysis/"
	@echo "                  vars: RAG_FAST=1 EVAL_OUTFILE=phase2"
	@echo "  make chat     - Ask the ReAct RAG agent (agent/rag_agent.py) a single question"
	@echo "                  vars: QUESTION=... (requires a materialized KB under semantic chunking)"
	@echo "  make test     - Run tests/rag/ (pytest -m PYTEST_MARK, default: phase0). Truncates kb_docs/kb_chunks per test."

migrate:
	$(PY) cli.py migrate

truncate:
	$(PY) cli.py truncate

ingest:
	$(PY) cli.py ingest \
		--base-dir "$(BASE_DIR)" \
		--strategy "$(STRATEGY)" \
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

eval:
	$(PY) cli.py eval $(if $(RAG_FAST),--fast) --outfile "$(EVAL_OUTFILE)"

chat:
	$(PY) cli.py chat "$(QUESTION)"

test:
	$(PY) -m pytest -m "$(PYTEST_MARK)" -v
