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

**Current state:** ingestion (`rag/db.py`, `rag/loaders.py`, `rag/ingest.py`, `sql/`, `cli.py`
migrate/ingest) is implemented, has been run end-to-end against the real Supabase DB, and has a
pytest suite (`tests/rag/`, phase0-marked) covering it. Still empty placeholder files:
`rag/search.py`, `eval/run.py`, `eval/queries.yaml`, `eval/experiments.yaml`. There is no linter
or build step configured yet — don't invent commands for tooling that doesn't exist. See
`README.md` for the full phase roadmap (ingestion → retrieval → generation → tone refinement) and
current position on it.

## Commands

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt   # setup
make migrate                                                            # apply sql/*.sql, in filename order
make ingest BASE_DIR=kb STRATEGY=fixed EMPRESA=empresa_x                # stage + chunk + embed
```

`make ingest` variables (see `Makefile`): `BASE_DIR` (default `kb`), `STRATEGY`
(`fixed`|`markdown`|`semantic`, default `fixed`), `EMPRESA`, `CLIENT_ID`, `CHUNK_SIZE` (800),
`CHUNK_OVERLAP` (200), `SKIP_STAGE=1`, `SKIP_CHUNKS=1`, `TRUNCATE=1` (destructive — truncates
`kb_docs`/`kb_chunks` first). `make ingest` is a thin wrapper over `cli.py ingest`; both accept
the same options (kebab-case flags on the CLI). `make help` lists both targets.

The ingest pipeline is also a LangGraph graph (`rag/ingest.py:graph`), runnable outside the CLI
via `langgraph dev` (uses `langgraph.json`, exposes it as `rag_ingest`; writes checkpoint state
to `.langgraph_api/`, currently not gitignored).

```bash
make test                          # pytest -m phase0 (default) against tests/rag/
make test PYTEST_MARK=phase1       # or any other marker from pytest.ini
```

`clean_db` (`tests/rag/conftest.py`) truncates `kb_docs`/`kb_chunks` before every test — don't
point `DB_*` at a database with data you care about while running the suite. To run a single
test: `.venv/bin/python -m pytest tests/rag/test_chunking.py::test_split_fixed_counts -v`.

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

### Database schema (`sql/`, applied in filename order — see `docs/database.md` for full detail)
Two-table design, staging separated from materialized/searchable chunks:
- `kb_docs` (`00_docs_staging.sql`) — raw ingested documents. Idempotent via
  `unique(source_path, source_hash)` — re-ingesting an unchanged file is a no-op.
- `kb_chunks` (`01_init.sql`) — chunked, embedded content. `embedding vector(1536)` (OpenAI
  `text-embedding-3-small` dimensions, generated via OpenRouter — see Environment) plus a
  generated `fts tsvector` column (`to_tsvector('portuguese', content)`) for full-text search.
  Idempotent via `unique(doc_path, chunk_ix)` — **known gap:** re-materializing a doc under a
  strategy that yields fewer chunks than a previous run leaves stale higher-`chunk_ix` rows
  behind, since upsert only touches indices the new run actually produces.
- Both tables carry `client_id uuid` / `empresa text` as tenant/grouping filters — this naming is
  kept intentionally (matches the course schema this was modeled on), not an oversight to "fix".
- `02_indexes.sql` — HNSW (`vector_cosine_ops`) on `embedding`, GIN on `fts`.
- `03_functions.sql` — retrieval logic lives in SQL, not application code:
  `kb_vector_search`, `kb_text_search`, `kb_hybrid_search` (Reciprocal Rank Fusion, fixed
  `K=60`, weights text=1.0/vector=1.5), and `kb_hybrid_union` (simple union, for comparison).
  All accept `p_client_id`/`p_empresa`/`p_chunking` filters and an optional `p_threshold` cutoff
  on vector similarity (`score = 1 - cosine_distance`).
- `sql/migrate.py` (`migrate()`) applies every `sql/*.sql` file in filename order, unconditionally
  — every statement is `create ... if not exists` / `create or replace`, so re-running is safe.

### Application layout (`rag/`)
Mirrors the pipeline stages; each file is independently swappable:
- `db.py` — `get_conn()`/`get_db_url()`, the only place DB connection details are assembled.
- `loaders.py` — three chunking strategies (`split_fixed`/`split_markdown`/`split_semantic`,
  dispatched via `split_text`): fixed-size (`RecursiveCharacterTextSplitter`), markdown-header-
  aware (`MarkdownHeaderTextSplitter`, ignores `chunk_size`/`chunk_overlap`), and semantic
  (`SemanticChunker`, embedding-guided, also ignores `chunk_size`/`chunk_overlap`). Each strategy
  fails explicitly (`RuntimeError`) rather than silently falling back if unavailable (e.g.
  semantic chunking without an embedder or `langchain-experimental`). See `docs/ingest.md` for a
  detailed comparison of the three, including observed chunk counts on the sample `kb/` data.
- `ingest.py` — two-phase ingestion: `stage_docs_from_dir` stages raw files into `kb_docs`
  (idempotent, filesystem read), `materialize_chunks_from_staging` chunks+embeds+upserts into
  `kb_chunks` (reads only from `kb_docs`, never the filesystem — re-chunking with a new strategy
  doesn't require re-staging). Also exposes `graph`, a two-node LangGraph `StateGraph`
  (`stage_docs -> chunk_docs`) wrapping the same two functions, referenced by both `cli.py` and
  `langgraph.json`. `get_embedder()` is the single place that owns the embeddings provider choice
  (OpenRouter).
- `search.py` (stub) — the retrieval function(s) calling the `sql/03_functions.sql` SQL
  functions, with optional HyDE query expansion and optional reranking (e.g. Cohere).

### `tests/rag/` — pytest suite
Deliberately mirrors `../ai-agent-sales/tests/rag/` (same layout, same `phase0`-`phase3` marker
scheme via `pytest.ini`, same fixture names in `conftest.py`), adapted where this repo's actual
code differs from that course project:
- `conftest.py` — `db_ready` (applies `sql/*.sql` via `sql.migrate.migrate()`, session-scoped,
  skips dependent tests if `DB_*` isn't configured/reachable), `clean_db` (autouse — truncates
  `kb_docs`/`kb_chunks` before every test), `require_embeddings` (skips unless
  `OPENROUTER_API_KEY` is set), `require_cohere`, `llm_judge_yesno` (routed through OpenRouter,
  same as `rag/ingest.get_embedder()`).
- **Ported (`phase0`, passing):** `test_kb_schema.py`, `test_kb_indexes.py`, `test_chunking.py`,
  `test_kb_ingest_graph.py`, `test_ingest_graph_two_nodes.py`, `test_kb_staging_smoke.py`,
  `test_kb_stage_then_chunk.py`, `test_kb_ingest_smoke.py`, `test_langgraph_registry.py`
  (unmarked). Adapted for real differences from the source repo: `app.rag.*` → `rag.*` imports;
  `IngestState` here has no `processed` alias, only `staged`/`chunked`; there's no standalone
  `ingest_dir()` helper (tests use `graph.invoke(...)` instead); `test_langgraph_registry.py`
  only asserts `rag_ingest` is registered (no `workflow`/`proposal_agent_v2` — those don't exist
  here).
- **Data files ported as-is:** `experiments.yaml`, `experiments.template.yaml`, `rag_queries.yaml`
  — same schema (`chunking`/`search_type`/`use_hyde`/`match_threshold`/`reranker` per experiment;
  `group`/`empresa`/`pergunta`/`expected`|`expected_all`/`answer` per query), same questions —
  they line up 1:1 with `kb/empresa_x`/`kb/empresa_y`, which are byte-identical to that repo's
  `tests/rag/data/`.
- **Not ported (needs `rag/search.py`/an agent, neither built yet):** `test_filters.py`,
  `test_retrieval_strategies.py`, `test_rrf_vs_union.py`, `test_rerank.py`,
  `test_retrieval_eval_llm.py`, `test_phase1_chunking_vector_only.py`,
  `test_phase1_semantic_qa.py`, `test_eval_dataset.py`, `test_experiments_runner.py`,
  `test_proposal_agent_v2.py`, `test_phase3_agent_dataset.py`. Port these once the code they
  exercise exists — they all import `app.rag.tools.kb_search_client` or similar, which has no
  equivalent here yet.

### `eval/`
Still just empty placeholder files (`queries.yaml`, `experiments.yaml`, `run.py`). Given
`tests/rag/experiments.yaml`/`rag_queries.yaml` now own that config, `eval/run.py`'s remaining
plausible role is a standalone report script (hit@k tables, etc.) consuming the same
`tests/rag/` YAML files — mirroring `../ai-agent-sales/scripts/rag_phase2_runner.py` — rather
than owning its own separate query/experiment definitions.

### Agent (planned)
This framework will eventually include a conversational/tool-calling RAG agent on top of
ingestion + retrieval + evaluation — not yet built, and not scaffolded here. When it lands, it
should import from `rag/search.py` rather than duplicating retrieval logic.

### `kb/`
Sample data only (`kb/empresa_x/`, `kb/empresa_y/`, each with `lead_reports/`, `policies/`,
`products/`, `services/` `.md` files) — placeholder content for exercising the pipeline, not
meant to be the permanent knowledge base. This and `tests/rag/rag_queries.yaml` are the parts
expected to change per use of the framework; `rag/` and `sql/` are the reusable engine.
