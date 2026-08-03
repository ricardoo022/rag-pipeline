# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A reusable RAG (retrieval-augmented generation) pipeline framework: Postgres/pgvector-backed
ingestion + hybrid retrieval + evaluation, meant to be reused across future projects (drop new
docs in `kb/`, write new eval questions, reuse the rest). It is a companion to the sibling repo
`../ai-agent-sales` (a LangGraph/LangChain course project) — this project follows that course's
RAG *mechanics* (staged ingestion, pluggable chunking, hybrid search, phased evaluation) but
deliberately drops everything CRM/sales-specific. Do not port CRM/lead/proposal code from that
repo here; if in doubt, treat `../ai-agent-sales` as reference material, not a source to copy
wholesale.

**Current state: early scaffold.** The schema (`sql/`) and DB connection (`rag/db.py`) are real.
Everything else — `rag/loaders.py`, `rag/ingest.py`, `rag/search.py`, `cli.py`, `eval/run.py` — is
an empty placeholder file, not yet implemented. There is no test suite, linter, or build step
configured yet; don't invent commands for tooling that doesn't exist.

## Environment

- Requires **Python 3.11+** (`python3.11`, not the system default `python3`/3.10 — several pinned
  deps like `ipython==9.5.0` require ≥3.11 and fail to resolve on 3.10).
- Virtualenv lives at `.venv/`. Install deps: `.venv/bin/pip install -r requirements.txt`.
- `requirements.txt` intentionally mirrors the full pinned dependency freeze from
  `../ai-agent-sales/requirements.txt` (minus the macOS-only `appnope`) rather than a minimal
  subset — keep it that way unless asked to trim it.
- Config lives in `.env` (gitignored; copy from `.env.example`). Required: `OPENROUTER_API_KEY`
  (embeddings, routed through OpenRouter's OpenAI-compatible `/api/v1/embeddings` rather than
  OpenAI directly — model id keeps the `openai/` prefix, e.g. `openai/text-embedding-3-small`,
  still 1536-dim so it matches `kb_chunks.embedding`), `DB_USER`/`DB_PASSWORD`/`DB_HOST`/`DB_PORT`/
  `DB_NAME`/`DB_SSLMODE`. Optional: `COHERE_API_KEY` (reranking), `LANGSMITH_API_KEY` (tracing).
- Database is Supabase Postgres with the `pgvector` extension. Connect via the **session pooler**
  (`*.pooler.supabase.com`, port 5432, user `postgres.<project-ref>`) — the direct
  `db.<project-ref>.supabase.co` host is IPv6-only and will hang from most local/WSL networks.
- This project and `../ai-agent-sales` are intended to point at the **same** Supabase database
  (shared `kb_docs`/`kb_chunks` data), not separate copies — keep schema changes compatible with
  both if both are in active use.

## Architecture

### Database schema (`sql/`, applied in filename order)
Two-table design, staging separated from materialized/searchable chunks:
- `kb_docs` (`00_docs_staging.sql`) — raw ingested documents. Idempotent via
  `unique(source_path, source_hash)` — re-ingesting an unchanged file is a no-op.
- `kb_chunks` (`01_init.sql`) — chunked, embedded content. `embedding vector(1536)` (OpenAI
  `text-embedding-3-small` dimensions) plus a generated `fts tsvector` column
  (`to_tsvector('portuguese', content)`) for full-text search. Idempotent via
  `unique(doc_path, chunk_ix)`.
- Both tables carry `client_id uuid` / `empresa text` as tenant/grouping filters — this naming is
  kept intentionally (matches the course schema this was modeled on), not an oversight to "fix".
- `02_indexes.sql` — HNSW (`vector_cosine_ops`) on `embedding`, GIN on `fts`.
- `03_functions.sql` — retrieval logic lives in SQL, not application code:
  `kb_vector_search`, `kb_text_search`, `kb_hybrid_search` (Reciprocal Rank Fusion, fixed
  `K=60`, weights text=1.0/vector=1.5), and `kb_hybrid_union` (simple union, for comparison).
  All accept `p_client_id`/`p_empresa`/`p_chunking` filters and an optional `p_threshold` cutoff
  on vector similarity (`score = 1 - cosine_distance`).

### Application layout (`rag/`)
Mirrors the pipeline stages; each file is independently swappable:
- `db.py` — `get_conn()`/`get_db_url()`, the only place DB connection details are assembled.
- `loaders.py` (stub) — chunking strategies: fixed-size, markdown-header-aware, semantic
  (embedding-guided). Match the course's convention of failing explicitly rather than silently
  falling back if a requested strategy is unavailable (e.g. semantic chunking without the right
  package/embeddings installed).
- `ingest.py` (stub) — two-phase ingestion: stage raw files into `kb_docs`, then materialize
  chunks into `kb_chunks` per strategy. Keep these phases separable so re-chunking with a new
  strategy doesn't require re-reading source files.
- `search.py` (stub) — the retrieval function(s) calling the `sql/03_functions.sql` SQL
  functions, with optional HyDE query expansion and optional reranking (e.g. Cohere).

### `eval/`
Where "does this actually retrieve well" gets answered — not a pytest suite. `queries.yaml` holds
hand-written questions with expected terms/answers; `experiments.yaml` defines a sweep
(chunking × search_type × HyDE × threshold); `run.py` (stub) is meant to report retrieval metrics
(hit@k) and optionally LLM-judge generated answers, in one runner rather than the split
pytest-tests-plus-parallel-runner-scripts pattern used in `../ai-agent-sales`.

### No agent (yet)
This framework does not include a conversational/tool-calling agent — the scope is retrieval
ingestion + evaluation only. If a chat agent is added later, it should import from `rag/search.py`
rather than duplicating retrieval logic.

### `kb/`
Empty by default — source `.md` documents to ingest go here. This and `eval/queries.yaml` are the
parts expected to change per use of the framework; `rag/` and `sql/` are the reusable engine.
