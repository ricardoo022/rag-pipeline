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
migrate/ingest), retrieval (`rag/search.py`, `cli.py search`), and a conversational RAG agent
(`agent/`, `cli.py chat`) are implemented, run end-to-end against the real Supabase DB, and
covered by the `tests/rag/` pytest suite (`phase0`-`phase3` markers). `eval/run.py` (+ `cli.py
eval`) is a standalone experiments report script sweeping retrieval-only experiments;
`eval/run_agent.py` (+ `cli.py eval-agent`) is its Phase 3 sibling — runs the full agent
(`agent/rag_agent.py`) over the same query dataset and LLM-judges the final answer instead of the
retrieved chunks. `eval/queries.yaml`/`eval/experiments.yaml` are intentionally still empty — that
config lives in `tests/rag/rag_queries.yaml`/`experiments.yaml` instead (see `eval/` below).
`test_proposal_agent_v2.py` from the sibling repo stays unported — it exercises the CRM-flavored
proposal agent, not the RAG-only one built here. `test_phase3_agent_dataset.py` **does** have an
equivalent here now (same filename, adapted to import `agent.rag_agent.graph` instead of
`app.rag.proposal_agent_v2` and to use the `seeded_kb_empresa_xy` fixture, since this repo's
`clean_db` truncates the KB before every test unlike the source repo's) — see "Agent (`agent/`)"
below. There is no linter or build step configured yet — don't invent commands for tooling that
doesn't exist. See `README.md` for the full phase roadmap (ingestion → retrieval → generation →
tone refinement) and current position on it.

## Commands

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt   # setup
make migrate                                                            # apply sql/*.sql, in filename order
make rag_phase0 BASE_DIR=kb EMPRESA=empresa_x                           # stage + chunk + embed, all 3 strategies
```

`make rag_phase0` variables (see `Makefile`): `BASE_DIR` (default `kb`), `STRATEGY`
(`fixed`|`markdown`|`semantic`, **default is all three** — `"fixed markdown semantic"`; pass e.g.
`STRATEGY=fixed` to materialize just one — stages once, then materializes each strategy under its
own `doc_path_prefix` so they don't collide on the `(doc_path, chunk_ix)` upsert key), `EMPRESA`,
`CLIENT_ID`, `CHUNK_SIZE` (800), `CHUNK_OVERLAP` (200), `SKIP_STAGE=1`, `SKIP_CHUNKS=1`,
`TRUNCATE=1` (destructive — truncates `kb_docs`/`kb_chunks` first). `make rag_phase0` is a thin
wrapper over `cli.py ingest`; both accept the same options (kebab-case flags on the CLI,
`--strategy` takes `nargs="+"`, always passed to the graph as `IngestState.strategies`). `make
help` lists both targets. The name mirrors `../ai-agent-sales`' `make rag_phase0` (same
staging+materialization concept), and the multi-strategy handling now mirrors that repo's
architecture too: `IngestState` carries a `strategies: List[str]` field and `rag/ingest.py`'s
`node_chunk` loops over it — same shape as `app/rag/ingest.py`'s `node_chunk` there — rather than
looping in `cli.py`. One deliberate difference from the sibling repo: there, `doc_path_prefix=s`
is applied unconditionally (even for a single strategy, whose default there is `semantic`); here
it's only applied when `len(strategies) > 1`, so a single-strategy run still writes
`doc_path == source_path` — this repo and `../ai-agent-sales` share the same Supabase KB tables,
and always-prefixing would orphan/duplicate rows already materialized under the old
no-prefix scheme by either repo. `node_chunk` still also accepts a singular `strategy` string
(back-compat for existing `graph.invoke({"strategy": ...})` callers, including `tests/rag/`).
Also unlike the sibling repo, there's no truncate-by-default or empresa-autodetect-from-subdirectory
("AUTO") mode — pass `TRUNCATE=1` explicitly if you want a clean re-run, and always pass `EMPRESA`
explicitly.

The ingest pipeline is also a LangGraph graph (`rag/ingest.py:graph`), runnable outside the CLI
via `langgraph dev` (uses `langgraph.json`, exposes it as `rag_ingest`; writes checkpoint state
to `.langgraph_api/`, currently not gitignored).

```bash
make search QUERY="garantia do Produto A" EMPRESA=empresa_x   # cli.py search, prints JSON
make rag_phase2_save_fast                                      # cli.py eval --fast -> eval/analysis/*.{json,csv}
make rag_phase3_save_fast                                      # cli.py eval-agent --fast -> eval/analysis/*.{json,csv}
make chat QUESTION="Qual a garantia do Produto A?"              # cli.py chat, one-shot ReAct answer
```

The agent needs a `semantic`-chunked, globally-queryable KB to be useful (its search tool fixes
`chunking=semantic`, no `empresa`/`client_id` filter) — e.g. `make rag_phase0 STRATEGY=semantic
EMPRESA=empresa_x` first.

```bash
make test                          # pytest -m phase0 (default) against tests/rag/
make test PYTEST_MARK=phase1       # or any other marker from pytest.ini
make rag_phase1                    # equivalent to: make test PYTEST_MARK=phase1
make rag_phase2_fast               # RAG_FAST=1 pytest -m phase2 (~10 questions, no CSV/JSON output)
make rag_phase2_extra              # pytest -m phase2, full question set
make rag_phase3_fast               # RAG_FAST=1 pytest -m phase3 (~10 questions, E2E agent, no CSV/JSON output)
make rag_phase3_extra              # pytest -m phase3, full question set
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
- `loaders.py` — five chunking strategies (`split_fixed`/`split_token`/`split_markdown`/
  `split_markdown_hybrid`/`split_semantic`, dispatched via `split_text`): fixed-size
  (`RecursiveCharacterTextSplitter`, chars), token-based (same splitter via
  `from_tiktoken_encoder`, `chunk_size`/`chunk_overlap` count tokens instead of chars),
  markdown-header-aware (`MarkdownHeaderTextSplitter`, ignores `chunk_size`/`chunk_overlap`),
  markdown-hybrid (header split first, then recursively sub-splits any section still over
  `chunk_size`), and semantic (`SemanticChunker`, embedding-guided, also ignores
  `chunk_size`/`chunk_overlap`). Each strategy fails explicitly (`RuntimeError`) rather than
  silently falling back if unavailable (e.g. semantic chunking without an embedder or
  `langchain-experimental`). See `docs/ingest.md` for a detailed comparison, including observed
  chunk counts on the sample `kb/` data.
- `ingest.py` — two-phase ingestion: `stage_docs_from_dir` stages raw files into `kb_docs`
  (idempotent, filesystem read), `materialize_chunks_from_staging` chunks+embeds+upserts into
  `kb_chunks` (reads only from `kb_docs`, never the filesystem — re-chunking with a new strategy
  doesn't require re-staging; optional `target_empresa`/`doc_path_prefix` let the same staged docs
  be materialized under multiple strategies into the same empresa without colliding on the
  `(doc_path, chunk_ix)` upsert key — used by the phase1 chunking-comparison test). Also exposes
  `graph`, a two-node LangGraph `StateGraph` (`stage_docs -> chunk_docs`) wrapping the same two
  functions, referenced by both `cli.py` and `langgraph.json`. `get_embedder()` is the single place
  that owns the embeddings provider choice (OpenRouter).
- `search.py` — `kb_search_client` (a LangChain `@tool`), calling the `sql/03_functions.sql`
  functions via `query_candidates`, with optional HyDE query expansion (`hyde()`, OpenRouter-routed
  `ChatOpenAI`) and optional reranking (`apply_rerank()`; `reranker="cohere"` needs
  `COHERE_API_KEY` + `langchain-cohere`, both already in `requirements.txt`). Filter policy:
  querying without `client_id`/`empresa`/`chunking` raises `RuntimeError`. Ported from
  `../ai-agent-sales/app/rag/tools.py`, adapted to `rag.db.get_conn`/`rag.ingest.get_embedder()`
  (OpenRouter) instead of direct OpenAI.

### `tests/rag/` — pytest suite
Deliberately mirrors `../ai-agent-sales/tests/rag/` (same layout, same `phase0`-`phase3` marker
scheme via `pytest.ini`, same fixture names in `conftest.py`), adapted where this repo's actual
code differs from that course project:
- `conftest.py` — `db_ready` (applies `sql/*.sql` via `sql.migrate.migrate()`, session-scoped,
  skips dependent tests if `DB_*` isn't configured/reachable), `clean_db` (autouse — truncates
  `kb_docs`/`kb_chunks` before every test), `require_embeddings` (skips unless
  `OPENROUTER_API_KEY` is set), `require_cohere`, `llm_judge_yesno` (routed through OpenRouter,
  same as `rag/ingest.get_embedder()`), `seeded_kb_empresa_xy` (stages+materializes
  `kb/empresa_x`/`kb/empresa_y` into empresa `"Empresa X"`/`"Empresa Y"` under all three chunking
  strategies — needed because, unlike `../ai-agent-sales`, `clean_db` here wipes `kb_chunks` before
  *every* test, so phase1/phase2 tests that query the shared `rag_queries.yaml` dataset can't rely
  on out-of-band pre-seeding via `make rag_phase0` the way the source repo's tests do).
- **Ported (`phase0`, passing):** `test_kb_schema.py`, `test_kb_indexes.py`, `test_chunking.py`,
  `test_kb_ingest_graph.py`, `test_ingest_graph_two_nodes.py`, `test_kb_staging_smoke.py`,
  `test_kb_stage_then_chunk.py`, `test_kb_ingest_smoke.py`, `test_langgraph_registry.py`
  (unmarked). Adapted for real differences from the source repo: `app.rag.*` → `rag.*` imports;
  `IngestState` here has no `processed` alias, only `staged`/`chunked`; there's no standalone
  `ingest_dir()` helper (tests use `graph.invoke(...)` instead); `test_langgraph_registry.py`
  asserts `rag_ingest` and `rag_agent` are registered (no `workflow`/`proposal_agent_v2` — those
  don't exist here).
- **Also ported, retrieval-dependent (`phase0`/`phase1`/`phase2`/unmarked):** `test_filters.py`
  (phase0), `test_phase1_chunking_vector_only.py` (phase0), `test_phase1_semantic_qa.py` (phase1),
  `test_retrieval_strategies.py`/`test_rrf_vs_union.py`/`test_rerank.py`/`test_experiments_runner.py`
  (phase2), `test_retrieval_eval_llm.py`/`test_eval_dataset.py` (unmarked, legacy — gated behind
  `RAG_PHASE2_LEGACY=1`, skipped otherwise). Same `app.rag.tools` → `rag.search`,
  `app.rag.ingest.ingest_dir` → `rag.ingest.graph.invoke(...)` adaptation as the phase0 set above,
  plus `require_openai` → `require_embeddings`; the three that query the shared
  `rag_queries.yaml` dataset without ingesting their own fixture data
  (`test_phase1_semantic_qa.py`, `test_eval_dataset.py`, `test_experiments_runner.py`) take the
  `seeded_kb_empresa_xy` fixture instead of assuming an already-populated KB.
- **Ported (`phase3`):** `test_phase3_agent_dataset.py` — runs `agent/rag_agent.py`'s ReAct graph
  end-to-end over `rag_queries.yaml` and LLM-judges only the final answer (not retrieved chunks);
  asserts `llm_yes >= 0.5`. Adapted from the source repo's file of the same name: imports
  `agent.rag_agent.graph` instead of `app.rag.proposal_agent_v2.graph`, and — like the other tests
  that query the shared dataset — takes `seeded_kb_empresa_xy` instead of the source test's
  skip-if-KB-empty check, since `clean_db` here always wipes `kb_chunks` first.
- **Data files ported as-is:** `experiments.yaml`, `experiments.template.yaml`, `rag_queries.yaml`
  — same schema (`chunking`/`search_type`/`use_hyde`/`match_threshold`/`reranker` per experiment;
  `group`/`empresa`/`pergunta`/`expected`|`expected_all`/`answer` per query), same questions —
  they line up 1:1 with `kb/empresa_x`/`kb/empresa_y`, which are byte-identical to that repo's
  `tests/rag/data/`.
- **Not ported (CRM-specific, out of scope for this repo):** `test_proposal_agent_v2.py` — exercises
  `../ai-agent-sales`' own `proposal_agent_v2` graph directly rather than through a reusable
  import, and pulls in CRM fixtures this repo doesn't have. It's about the *sales* agent, not the
  RAG-only one built here.

### `eval/`
`run.py` (+ `cli.py eval`) is a standalone experiments report script: sweeps
`tests/rag/experiments.yaml` x `rag_queries.yaml` over the already-materialized KB via
`rag.search.kb_search_client`, and writes CSV/JSON to `eval/analysis/` (gitignored) — mirrors
`../ai-agent-sales/scripts/rag_phase2_runner.py`, adapted to OpenRouter-routed embeddings/LLM
judge. `run_agent.py` (+ `cli.py eval-agent`, `make rag_phase3_save*`) is its Phase 3 sibling —
same CSV/JSON-to-`eval/analysis/` idea, but runs `agent/rag_agent.py`'s graph end-to-end per
question and LLM-judges the final answer text instead of sweeping retrieval experiments; mirrors
`../ai-agent-sales/scripts/rag_phase3_runner.py` (writes to `eval/analysis/`, not
`tests/rag/analysis/`, and is wired through `cli.py` like `run.py` is, rather than being a
standalone script only invoked directly). `queries.yaml`/`experiments.yaml` in this directory are
intentionally still empty — that config lives in `tests/rag/rag_queries.yaml`/`experiments.yaml`
instead, which both `run.py` and `run_agent.py` read directly rather than owning their own
separate definitions.

### Agent (`agent/`)
A conversational, tool-calling RAG agent — no CRM/proposal tools, just KB search — imports
`rag.search.kb_search_client` rather than duplicating retrieval logic:
- `react.py` — `create_react_executor(model, tools, prompt=...)`, a generic two-node LangGraph
  loop (`agent -> tools -> agent`, looping until the model stops requesting tool calls). Not
  RAG-specific; reused as-is.
- `rag_agent.py` — the one KB-search tool (`kb_search`, fixed `chunking=semantic`/
  `search_type=hybrid_rrf`/`reranker=none`, only `match_threshold` is model-adjustable), an
  OpenRouter-routed `ChatOpenAI` (`RAG_AGENT_LLM_MODEL`, default `openai/gpt-4.1-mini`), and a
  system prompt that instructs grounding-only answers with `doc_path#chunk_ix` citations. Exposes
  `graph`, registered in `langgraph.json` as `rag_agent` and driven by `cli.py chat`/`make chat`.
  Ported from `../ai-agent-sales/app/rag/proposal_agent_v2.py` (already CRM-tool-free — its own
  docstring calls it an "isolated RAG-only test version") and `app/agent/new_react.py`, stripped
  of the "proposal agent" framing and adapted to OpenRouter. The KB-search tool passes no
  `empresa`/`client_id` filter (only the fixed `chunking`), so it searches across all empresas
  that have semantic chunks — same behavior as the source.
- Deliberately not ported: `app/agent/workflow.py`, `tools.py`, `helpers.py`, `prompt.py` (the
  CRM/lead/proposal sales agent) and `app/rag/proposal_agent_v2.py`'s own CRM siblings — out of
  scope per "What this is" above.

### `kb/`
Sample data only (`kb/empresa_x/`, `kb/empresa_y/`, each with `lead_reports/`, `policies/`,
`products/`, `services/` `.md` files) — placeholder content for exercising the pipeline, not
meant to be the permanent knowledge base. This and `tests/rag/rag_queries.yaml` are the parts
expected to change per use of the framework; `rag/` and `sql/` are the reusable engine.
