# Ingest (Phase 0)

Two-phase pipeline that turns `.md` files under `kb/` into searchable rows in `kb_chunks`:
stage raw files into `kb_docs`, then chunk + embed them into `kb_chunks`. Implemented as a
LangGraph graph (`rag/ingest.py:graph`) so the same pipeline runs from the CLI, `make`, or
LangGraph Studio.

## Pipeline

```
kb/*.md --[stage_docs]--> kb_docs --[chunk_docs]--> kb_chunks
```

1. **`stage_docs`** (`stage_docs_from_dir`, `rag/ingest.py`) — reads every `.md` under a base
   dir, inserts one row per file into `kb_docs` with a `sha256` of its content. Idempotent via
   `unique(source_path, source_hash)`: re-running on unchanged files is a no-op; an edited file
   inserts a new staged version rather than overwriting.
2. **`chunk_docs`** (`materialize_chunks_from_staging`, `rag/ingest.py`) — reads rows back out
   of `kb_docs` (optionally filtered by `empresa`/`client_id`/`path_prefix`), splits each
   document's `content` per the requested chunking strategy, embeds the resulting chunks, and
   upserts them into `kb_chunks`. Reads only from `kb_docs`, never the filesystem — re-chunking
   with a different strategy doesn't require re-staging.

Both phases go through `rag/db.get_conn()`. Splitting on two phases like this means "reprocess
with a new chunking strategy" and "reingest changed source files" are independent operations.

## Chunking strategies (`rag/loaders.py`)

`split_text(text, strategy, embedder=None, chunk_size=800, chunk_overlap=200)` dispatches to one
of the three splitters below and returns `(chunks, resolved_strategy)`. An invalid or
unavailable strategy **raises** (`RuntimeError`) rather than silently falling back to `fixed` —
e.g. calling `semantic` without an embedder, or without `langchain-experimental` installed, is a
hard error.

All three also get a blank-chunk filter (`[c for c in chunks if c.strip()]`) — `SemanticChunker`
in particular emits a trailing empty-string chunk on some inputs, which some embeddings
endpoints (OpenRouter) reject outright as invalid input.

The resolved strategy is stored as `meta.chunking` on each `kb_chunks` row, which is what
`sql/03_functions.sql`'s `p_chunking` filter matches against — so all three can be materialized
side by side (as long as they don't collide on `(doc_path, chunk_ix)`, see the known gap below)
and compared at query time by passing `p_chunking`.

### `fixed` — `split_fixed` / `RecursiveCharacterTextSplitter`

Pure character-count splitting: cuts at `chunk_size` characters, walking down a hierarchy of
separators (`"\n\n"`, `"\n"`, `" "`, `""`) so it prefers breaking on a paragraph or line boundary
over slicing mid-word when one is available nearby. `chunk_overlap` repeats that many trailing
characters of one chunk at the start of the next, so a fact or sentence split across a chunk
boundary is still fully present in at least one chunk.

- No structural awareness at all — will happily cut a header away from its body text.
- Cheapest and fastest: no LLM/embedding calls, output size is deterministic from `chunk_size`.
- Chunk count scales predictably with document length: roughly
  `len(text) / (chunk_size - chunk_overlap)`.
- Good default when you don't know/trust the source formatting, or want a stable baseline to
  compare the other two strategies against.

### `markdown` — `split_markdown` / `MarkdownHeaderTextSplitter`

Splits purely on markdown header boundaries (`#`, `##`, `###`) — one chunk per section, with the
header text itself attached as chunk metadata by the underlying splitter. **`chunk_size` and
`chunk_overlap` are accepted by the function signature but not applied** — a section under one
header becomes exactly one chunk no matter how long or short it is; there's no secondary
size-based split.

- Preserves whatever semantic unit the author already encoded via headers — a `## Garantia e
  Suporte` section stays intact as one chunk instead of being cut arbitrarily.
- Chunk count is driven by header density, not `chunk_size`: a doc with many small `##`/`###`
  sections produces many small chunks; a doc with few headers (or long stretches before the
  first header) produces few, large ones. On `kb/empresa_x`'s 5 files this strategy produced
  **34 chunks**, nearly 3x `fixed`'s **12**, because those docs are broken into many short `##`
  sections.
- Risk: a very long section under a single header still becomes one (potentially huge) chunk —
  and a short, context-free header (e.g. a one-line "Observações" section) becomes its own
  chunk with little content to search on.
- No embedding calls needed — same cost profile as `fixed`.

### `semantic` — `split_semantic` / `SemanticChunker` (langchain-experimental)

Embeds each sentence, measures the semantic distance between consecutive sentence embeddings,
and cuts wherever that distance is a statistical outlier — `breakpoint_threshold_type=
"interquartile"` means a cut happens where the jump exceeds a threshold based on the
interquartile range of all consecutive-sentence distances in the document. `chunk_size`/
`chunk_overlap` don't apply here either; boundaries are entirely content-driven.

- Requires an `embedder` and makes its own embedding calls *just to find breakpoints*, on top of
  the calls `materialize_chunks_from_staging` later makes to embed the resulting chunk content
  — the most expensive strategy of the three, and the only one with an external dependency that
  can fail independently of the DB/staging step.
- Produces the fewest, most topically-coherent chunks when a document doesn't shift topic much
  sentence-to-sentence — on the same 5 `kb/empresa_x` files this strategy produced only **5
  chunks total** (essentially one chunk per file), versus 12 (`fixed`) and 34 (`markdown`). For
  short, topically-uniform docs like these sample files, that's a sign the technique is
  under-splitting rather than finding meaningful internal boundaries — it's better suited to
  longer, more topically varied source documents.
- Chunk sizes are unpredictable and can vary widely within the same run, since nothing bounds
  how long a "no breakpoint found" run of sentences can get.

### Choosing one

`fixed` is the cheap, predictable baseline; `markdown` is free and structure-aware but only as
good as the source document's own headers; `semantic` is the most content-aware but costs extra
embedding calls and can over- or under-split depending on how topically uniform the source text
is. This is exactly the kind of tradeoff `eval/experiments.yaml` (sweep of chunking ×
search_type × HyDE × threshold, still a stub) is meant to measure empirically rather than settle
by inspection.

## Embeddings

`rag/ingest.get_embedder()` builds an `OpenAIEmbeddings` client pointed at **OpenRouter's**
OpenAI-compatible endpoint rather than OpenAI directly:

- `openai_api_base = "https://openrouter.ai/api/v1"`
- `model = "openai/text-embedding-3-small"` (OpenRouter requires the `openai/` provider prefix)
- `openai_api_key` read from `OPENROUTER_API_KEY`

Still resolves to 1536-dim vectors, so it's a drop-in match for `kb_chunks.embedding
vector(1536)` — no schema change needed. `embed_texts()` (used for chunk content) and
`materialize_chunks_from_staging()`'s `semantic`-strategy embedder both go through this one
function/client, so there's a single place that owns the provider choice.

## Upsert (`upsert_chunks`)

Writes are idempotent on `(doc_path, chunk_ix)` — `on conflict ... do update`. Embeddings are
passed as a pgvector literal string (`vec_to_literal`: `"[0.1,0.2,...]"`) cast to `vector(1536)`
in the query.

**Known gap:** re-chunking a document under a strategy that produces *fewer* chunks than the
previous run leaves the excess higher-`chunk_ix` rows behind with stale content/`meta`, since
the upsert only touches indices the new run actually produces. Not handled yet — relevant once
`eval/run.py` starts sweeping strategies against the same `kb/` files.

## LangGraph graph (`rag/ingest.py`)

```python
class IngestState(TypedDict, total=False):
    base_dir: str
    strategy: str
    client_id: Optional[str]
    empresa: Optional[str]
    chunk_size: Optional[int]
    chunk_overlap: Optional[int]
    skip_stage: bool
    skip_chunks: bool
    staged: int
    chunked: int
```

`compile_graph()` builds a two-node `StateGraph`: `stage_docs -> chunk_docs`, entry at
`stage_docs`, finish at `chunk_docs`. Each node returns a partial state update (`{"staged": n}` /
`{"chunked": n}`), which LangGraph merges into the running state. `skip_stage`/`skip_chunks` let
either phase be skipped without branching the graph shape. The module-level `graph` is the
compiled instance, referenced from both `cli.py` and `langgraph.json`.

`langgraph.json` exposes it as `rag_ingest` for `langgraph dev` / LangGraph Studio:

```json
{
  "dependencies": ["."],
  "env": ".env",
  "graphs": { "rag_ingest": "rag/ingest.py:graph" }
}
```

## Running it

### `make ingest`

```
make ingest BASE_DIR=kb/empresa_x STRATEGY=fixed EMPRESA=empresa_x
```

Variables (all optional, see `Makefile`):

| var | default | meaning |
|---|---|---|
| `BASE_DIR` | `kb` | directory of `.md` files to stage |
| `STRATEGY` | `fixed` | `fixed` \| `markdown` \| `semantic` |
| `EMPRESA` | *(empty)* | tags staged docs/chunks |
| `CLIENT_ID` | *(empty)* | tags staged docs/chunks (uuid) |
| `CHUNK_SIZE` | `800` | only used by `fixed` |
| `CHUNK_OVERLAP` | `200` | only used by `fixed` |
| `SKIP_STAGE` | *(unset)* | set to `1` to only re-materialize chunks from existing `kb_docs` |
| `SKIP_CHUNKS` | *(unset)* | set to `1` to only stage, no chunking/embedding |
| `TRUNCATE` | *(unset)* | set to `1` to `truncate` `kb_docs`/`kb_chunks` first (destructive) |

### `cli.py` directly

```
.venv/bin/python cli.py ingest --base-dir kb/empresa_x --strategy fixed --empresa empresa_x
```

Same flags as the `make ingest` variables, kebab-cased (`--chunk-size`, `--skip-stage`, ...).
`make ingest` is a thin wrapper around this.

### Config required

- `OPENROUTER_API_KEY` in `.env` (embeddings — see above).
- `DB_*` in `.env` (see `docs/database.md`).
- `make migrate` must have been run at least once (`kb_docs`/`kb_chunks`/functions/indexes need
  to exist).

## Files

| file | role |
|---|---|
| `rag/loaders.py` | chunking strategies (`split_fixed`/`split_markdown`/`split_semantic`/`split_text`), `.md` file discovery |
| `rag/ingest.py` | staging, materialization, upsert, embeddings, the LangGraph graph |
| `cli.py` | `ingest` subcommand — parses flags, invokes `graph.invoke(...)` |
| `Makefile` | `make ingest` — ergonomic wrapper over `cli.py ingest` |
| `langgraph.json` | exposes `graph` as `rag_ingest` for LangGraph Studio/`langgraph dev` |
