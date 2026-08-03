.PHONY: help migrate ingest

PY := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else command -v python; fi)

BASE_DIR ?= kb
STRATEGY ?= fixed
EMPRESA ?=
CLIENT_ID ?=
CHUNK_SIZE ?= 800
CHUNK_OVERLAP ?= 200

help:
	@echo "Available commands:"
	@echo "  make migrate  - Apply sql/*.sql against the configured database"
	@echo "  make ingest   - Stage kb/*.md into kb_docs, then chunk+embed into kb_chunks"
	@echo "                  vars: BASE_DIR=kb STRATEGY=fixed|markdown|semantic EMPRESA=... CLIENT_ID=..."
	@echo "                        CHUNK_SIZE=800 CHUNK_OVERLAP=200 SKIP_STAGE=1 SKIP_CHUNKS=1 TRUNCATE=1"

migrate:
	$(PY) cli.py migrate

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
