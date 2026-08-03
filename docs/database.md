# Database

Postgres + [pgvector](https://github.com/pgvector/pgvector), hosted on Supabase. Two tables:
raw documents staged in `kb_docs`, then chunked/embedded into `kb_chunks`. Retrieval logic
(vector, full-text, hybrid) lives in SQL functions rather than application code.

## Connecting

Use the **session pooler**, not the direct host:

- Direct (`db.<project-ref>.supabase.co:5432`) is **IPv6-only** — unreachable from most local/WSL
  networks (`Network is unreachable`).
- Session pooler (`aws-0-<region>.pooler.supabase.com:5432`, user `postgres.<project-ref>`)
  supports IPv4 and is what `.env` should point at.

Found in Supabase dashboard → **Project Settings → Database → Connection Pooling** (Session mode).

`.env` fields (see `.env.example`): `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`
(`postgres` — Supabase doesn't let you rename the default database), `DB_SSLMODE` (`require`).

`rag/db.py` builds the connection string from these; every DB-touching script goes through
`get_conn()`/`get_db_url()` there.

## Applying the schema

```
make migrate
```

Runs `sql/*.sql` in filename order via `sql/migrate.py`. Every statement is
`create ... if not exists` / `create or replace`, so re-running is always safe — no separate
"down" migrations. `01_init.sql` also runs `create extension if not exists vector;`, so pgvector
gets enabled automatically on first run (it's on Supabase's allowed extension list — no manual
dashboard step needed).

## Schema

### `kb_docs` — staging (`00_docs_staging.sql`)
Raw ingested files, one row per document version.

| column | type | notes |
|---|---|---|
| `source_path` | text | path/identifier of the source file |
| `source_hash` | text | content hash |
| `content` | text | raw document text |
| `mime_type` | text | default `text/markdown` |
| `meta` | jsonb | free-form metadata |
| `client_id` | uuid, null | optional tenant/project tag — see below |
| `empresa` | text, null | optional tenant/project tag — see below |

`unique (source_path, source_hash)` makes staging idempotent: re-ingesting an unchanged file is a
no-op; a changed file (new hash) inserts a new row rather than overwriting.

### `kb_chunks` — materialized, searchable (`01_init.sql`)
Chunked + embedded content, what search actually queries.

| column | type | notes |
|---|---|---|
| `doc_path` | text | which document this chunk came from |
| `chunk_ix` | int | chunk position within the document |
| `content` | text | chunk text |
| `embedding` | `vector(1536)` | OpenAI `text-embedding-3-small` |
| `fts` | tsvector, generated | `to_tsvector('portuguese', content)`, stored, auto-maintained |
| `meta` | jsonb | e.g. `{"chunking": "fixed"\|"markdown"\|"semantic"}` |
| `client_id` | uuid, null | optional tenant/project tag |
| `empresa` | text, null | optional tenant/project tag |

`unique (doc_path, chunk_ix)` makes materialization idempotent (upsert on re-chunk).

### `client_id` / `empresa`
Both nullable, both optional. They exist for segmenting the knowledge base if you ever want
multiple projects/domains in the same database — every search function accepts them as optional
filters, defaulting to `null` (no filtering). For solo/single-project use, leave them `null` and
ignore them entirely; nothing requires filling them in.

### Indexes (`02_indexes.sql`)
- `ix_kb_chunks_embedding_hnsw` — HNSW on `embedding` (`vector_cosine_ops`, `m=16`,
  `ef_construction=64`) for fast approximate vector search.
- `ix_kb_chunks_fts_gin` — GIN on `fts` for full-text search.
- Plain B-tree indexes on `(doc_path, chunk_ix)`, `empresa`, `client_id` on both tables.

## Search functions (`03_functions.sql`)

All take `p_k` (limit), optional `p_threshold`, and optional `p_client_id`/`p_empresa`/`p_chunking`
filters. Score is always `1 - cosine_distance` for the vector component (higher = more similar).

- **`kb_vector_search`** — pure vector similarity (`embedding <=> query_embedding`).
- **`kb_text_search`** — pure full-text search (`plainto_tsquery('portuguese', ...)`,
  `ts_rank_cd` score).
- **`kb_hybrid_search`** — vector + text combined via **Reciprocal Rank Fusion**
  (fixed `K=60`, weights: vector `1.5`, text `1.0`). This is the default/recommended search mode.
- **`kb_hybrid_union`** — simple union of vector + text candidates, no RRF. Kept for comparing
  against `kb_hybrid_search` when evaluating retrieval strategies, not for normal use.

`kb_vector_search`/`kb_hybrid_search` take the query embedding as a `text` literal
(`"[0.1,0.2,...]"`), cast to `vector(1536)` inside the function.
