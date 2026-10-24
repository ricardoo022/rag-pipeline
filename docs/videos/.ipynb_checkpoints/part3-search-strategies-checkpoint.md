# Part 3 — Search Strategies (plan)

Working title: *"Hybrid search: why your RAG needs two searches, not one"*
Series position: Part 3. Part 1 = overview + client management. Part 2 = chunking (search
deliberately cut). Part 3 (this) = **search strategies** — how the question turns into a set of
chunks. Part 4 = evaluators (LLM-as-judge on the agent's final answer).

Scope: the **four `search_type` values** the pipeline supports — `vector`, `text`, `hybrid_rrf`,
`hybrid_union`. HyDE and reranking are named as *add-ons* at the end but not taught in depth
(they get their own short segment or a later part).

---

## The idea in one paragraph

Chunking decided what is *possible* to retrieve. Search decides what actually comes back for a
given question. There are two fundamentally different ways to search: by **meaning** (vector) and
by **exact words** (full-text). Each has a blind spot the other covers — vector fumbles exact
numbers and codes, text misses anything phrased differently. So you run both and merge the two
result lists into one. *How* you merge them is the whole lesson: the naive merge (`union` — sort
both lists by their raw scores) is broken because the two scores aren't the same kind of number;
the right merge (`RRF` — throw the scores away, use only rank position) rewards the chunks that
*both* searches agreed on. We show all four on our own KB and land on the same thesis as Part 2:
you pick this with a number, not an opinion.

## What the viewer should be able to do afterwards

- Explain vector search: embed the question, measure cosine distance to every chunk vector,
  return the closest. Know that a small distance = high similarity.
- Explain text search: the question becomes a set of stemmed words ANDed together, matched
  against a precomputed `fts` column. Know why one missing word returns *nothing*.
- Say when each one wins: vector for paraphrase/meaning, text for exact tokens (numbers, codes,
  names).
- Explain hybrid search: run both, merge into one list.
- Explain the difference between `RRF` (merge by rank) and `union` (merge by score), and why
  RRF is the one to use.
- Run the eval in this repo and read the `hit@1 / hit@5 / judge_yes` table across search types.

---

## The four searches

| `search_type` | Matches on | SQL function | Blind spot |
|---|---|---|---|
| `vector` | meaning (embedding cosine distance) | `kb_vector_search` (`sql/03_functions.sql:5`) | exact numbers, codes, rare tokens |
| `text` | exact words (stemmed, full-text) | `kb_text_search` (`sql/03_functions.sql:38`) | any phrasing the document doesn't use |
| `hybrid_rrf` / `hybrid` | both, merged by **rank** | `kb_hybrid_search` (`sql/03_functions.sql:71`) | — (this is the one to ship) |
| `hybrid_union` | both, merged by **raw score** | `kb_hybrid_union` (`sql/03_functions.sql:115`) | text half is effectively ignored |

Dispatch for all four: `rag/search.py` → `query_candidates()` (lines 19–83). The Python side only
embeds the query and calls the SQL — all retrieval logic lives in `sql/03_functions.sql`.

The agent (`agent/rag_agent.py`) hard-codes `hybrid_rrf`. `text` is **never** run alone in the
eval — only as the keyword half inside hybrid.

---

## Segment 1 — Vector search (~3 min)

**Talk (Excalidraw + code):**
- At ingestion, every chunk's text was turned into a vector (1536 numbers) by the embedding
  model and stored in `kb_chunks.embedding`.
- At query time, the question goes through the *same* model → a query vector.
- Postgres compares the query vector to every chunk vector with **cosine distance** — the
  `<=>` operator that pgvector adds (`sql/03_functions.sql:29`, `:33`). It measures the *angle*
  between two arrows: same direction → distance 0, unrelated → distance ~1.
- `order by b.embedding <=> q.qvec asc` (line 33) — closest first. `limit p_k` — take the top
  few. That `order by` is the whole decision: **closer meaning wins.**
- The reported `score` is `1 - distance` (line 29), so ~1.0 = near-identical meaning.
- `p_threshold` (line 32): keep only chunks with `similarity >= threshold`. Without it, vector
  search *always* returns `k` rows even if the best one is garbage. This cutoff exists only in
  vector search — text search has no equivalent.

**The weakness to state out loud:** it matches meaning, not exact strings. `"99.9%"`, `"99.5%"`,
`"99.99%"` all land in nearly the same place — the model knows "a reliability percentage" but not
*which* one. Same for `30/60`, dates, versions, SKUs. Whenever the exact value *is* the answer,
vector search is a coin flip.

---

## Segment 2 — Text search (~2 min, keep it short)

**Talk (Excalidraw + code):** `kb_text_search`, `sql/03_functions.sql:38`.
- `plainto_tsquery('english', question)` (line 58) does four things: tokenize → drop stopwords
  ("what", "is", "the") → **stem** each word ("warranties" → `warranti`) → join them all with
  **AND** (`&`). `"What is the SLA for Service B?"` → `'sla' & 'servic' & 'b'`.
- `b.fts` is a **generated column** (`sql/01_init.sql`) — the chunk's `content` run through the
  same tokenize/stem pipeline at ingestion, stored as a `tsvector` (a bag of stemmed words). The
  search runs on `fts`, not on the raw text.
- `b.fts @@ q.tsq` (line 64) asks one thing: **are all the query's words present in this
  chunk's bag?** Extra words in the chunk don't matter — only missing ones. Miss one → the
  whole formula is false → chunk dropped.
- `ts_rank_cd(b.fts, q.tsq)` (line 65) ranks the survivors by how often the words appear and how
  close together. It's a small unbounded float — **not** on the same scale as vector's score.
- Draw the **inverted index**: `"warranty" → chunk 2, chunk 9` / `"sla" → chunk 4`. That one
  picture is the whole mechanism.

**The weakness to demo:** because every term is ANDed, a question with one word that's in no
chunk (`"what percentage is guaranteed"` → `'uptim' & 'percentag' & 'guarante'`) matches
**nothing** — an empty list, not bad results. On our 5-paragraph KB this happens constantly.

**The strength:** exact tokens. If the user types `30/60`, the chunk with `30/60` matches,
exactly, every time — no blurring.

**Framing:** text search is a *specialist*, not a general-purpose search. That's why it's never
run alone — only next to vector, inside hybrid.

---

## Segment 3 — Hybrid search, no fusion talk yet (~1 min)

Just the concept: run vector *and* text, get two ranked lists, merge into one. Show `hybrid_rrf`
landing — the paraphrase question and the exact-token question *both* come back right, where each
failed under one of the singles. Takeaway at this point: **use both, it beats either one.** Most
people can stop here.

---

## Segment 4 — Fusion: RRF vs union (~4 min, the core)

Both hybrid functions run the same two searches. They differ in **one step** — how the two lists
become one.

### RRF — `kb_hybrid_search` (`sql/03_functions.sql:71`)

The trick in one sentence: **ignore the scores, use only the position.** (Vector's top score is
~0.82, text's is ~0.03 — different kinds of number, you can't sort a pile by them. But "1st
place" means the same thing in both lists.)

Four CTEs:
1. `v` (lines 87–90) — run vector search, `row_number() over (order by score desc) as rnk_v` →
   tag each result with its position. Score is now discarded.
2. `t` (lines 91–94) — same for text → `rnk_t`.
3. `u` (lines 95–102) — `full outer join` on `(doc_path, chunk_ix)`: a chunk in both lists gets
   one row with both ranks; a chunk in only one list has the other rank `NULL`. Nobody is
   dropped.
4. `r` (lines 103–107) — **the fusion, line 105:**
   ```sql
   coalesce(1.5 / (60 + rnk_v), 0) + coalesce(1.0 / (60 + rnk_t), 0) as score
   ```
   For each list the chunk is in, add `weight / (60 + rank)`. Missing from a list → that term is
   `NULL` → `coalesce` makes it `0`. **The `+` is the fusion** — the exact point where the two
   searches become one number.
5. `order by score desc, limit p_k` (lines 108–111).

The formula, `weight / (K + rank)`, K = **60**:

| rank | 1/(60+rank) |
|---|---|
| 1 | 0.0164 |
| 2 | 0.0161 |
| 5 | 0.0154 |

- **Reciprocal** — higher rank contributes more, but the gap between #1 and #2 is tiny. K=60
  flattens it so neither list steamrolls the other. (60 is the value from the 2009 RRF paper —
  the literal `60` in the code.)
- **Additive** — a chunk in *both* lists gets both contributions summed, so a chunk that was #3
  in each beats a chunk that was #1 in one and absent from the other. **Agreement wins.**

**The weights** (`1.5` vector, `1.0` text, line 105) are hardcoded literals — comment on line 70
says so ("KISS"). Conceptually they're a dial: only the *ratio* matters (scaling both changes
nothing), useful range is ~1:1 to 2:1, past ~3:1 the lighter search stops mattering. To actually
tune them you'd add `p_w_vector` / `p_w_text` params and thread them through
`query_candidates` → `kb_search_client` → `experiments.yaml` — that plumbing doesn't exist yet.

### Union — `kb_hybrid_union` (`sql/03_functions.sql:115`)

The naive merge:
```sql
u as (select * from v union select * from t)
select ... from u order by score desc limit p_k
```
Pile both lists together, sort by the `score` column. But `score` isn't the same currency:
vector rows are ~0.7–0.9, text rows are ~0.02–0.1. Sort that mixed pile and the vector rows sit
on top no matter how good the text hits were. `hybrid_union` is effectively *"vector results,
with text leftovers stapled underneath"* — and a chunk both searches found appears **twice**
(two rows, two different scores; `UNION` only dedupes identical rows).

### Worked example (illustrative numbers)

Query: *"Confirm the 30/60 payment terms"* — the answer is in the **commercial** chunk.

**Vector returned:** `overview (0.79)`, `included (0.77)`, `notes (0.75)`, **`commercial (0.74)`**,
`joao (0.73)` — the answer is buried at #4.
**Text returned:** **`commercial (0.11)`**, `maria (0.05)` — text nailed the answer at #1.

**WITHOUT fusion (union)** — sort the combined pile by score:
```
overview 0.79 · included 0.77 · notes 0.75 · commercial 0.74 · joao 0.73   <- top 5, all vector
commercial 0.11 (dup) · maria 0.05                                          <- text, below the fold
```
Answer at #4. Text found it at #1 but 0.11 < 0.73, so text's opinion is thrown away. You just
ran vector search.

**WITH fusion (RRF)** — `1.5/(60+vec_rank) + 1.0/(60+text_rank)`:

| chunk | vec rank | text rank | points |
|---|---|---|---|
| **commercial** | 4 | 1 | 1.5/64 + 1.0/61 = **0.0398** |
| overview | 1 | – | 1.5/61 = 0.0246 |
| included | 2 | – | 1.5/62 = 0.0242 |
| notes | 3 | – | 1.5/63 = 0.0238 |
| joao | 5 | – | 1.5/65 = 0.0231 |

Answer jumps to **#1** — it's the only chunk *both* searches ranked, and the two contributions
added together beat five chunks that only vector liked.

**One-line takeaway:** without fusion, the search with the bigger raw numbers wins and the other
is dead weight. With fusion, the chunk both searches agreed on wins.

### The stale-comment beat

`sql/03_functions.sql:2` says *"evitamos RRF no banco; híbrido = união simples de candidatos"* —
that comment is stale, `kb_hybrid_search` right below it is pure RRF. Same energy as the markdown
header-drop in Part 2: you catch these by reading the code, not the comments.

---

## Segment 5 — The add-ons: HyDE and rerank (~3 min, shown in the notebook)

Both are orthogonal to the four search types: pick 1 of `vector`/`text`/`hybrid_rrf`/
`hybrid_union`, then optionally bolt either of these on top. Neither is a search method by
itself. Notebook cells 22–30 (`docs/videos/search.ipynb`) demo both live.

- **HyDE** (`use_hyde`, `rag/search.py:144`; applied at `:187–190`) — before searching, an LLM
  writes a fake hypothetical answer paragraph and you embed *that* instead of the question.
  Fixes the *question*. Only affects the vector side; silently ignored for `search_type="text"`
  — the notebook proves this with a before/after cell (23–28: print the generated paragraph,
  compare vector with/without HyDE on `Q_MEANING`, then show text search is unaffected).
  Cost: +1 LLM call per query, and it's non-deterministic — the ranking can shift run to run.
- **Rerank** (`reranker="cohere"`, `apply_rerank` — `rag/search.py:86`) — after searching, pull
  `rerank_candidates` (default 24) candidates and let a Cohere cross-encoder re-score each
  `(query, chunk)` pair jointly, keep the top `k`. Fixes the *order*, not recall — it can only
  reorder what retrieval already found, never pull in a chunk that was missed entirely. Needs
  `COHERE_API_KEY` + `langchain-cohere`; the notebook wraps the call in `try/except` since a
  trial key can 429 (same handling as `eval/run.py`'s `skipped_reason = "cohere_429"`).

---

## Segment 6 — How we test it (~3 min)

Transition: *"That was two questions, by eye. Same trap as chunking — two examples isn't a
result."*

- `tests/rag/experiments.yaml` — the sweep config. Each row is one combination of `chunking` ×
  `search_type` × `use_hyde` × `match_threshold` × `reranker`. For the shoot: hold
  `chunking: semantic` on every row, move only `search_type`.
- `eval/run.py` (`cli.py eval`) — the **same harness as Part 2**, just a different swept column.
  Runs every experiment row × every question in `tests/rag/rag_queries.yaml`.
- Metrics (same three as Part 2):
  - **hit@1** — does the expected string appear in the top-1 chunk?
  - **hit@5** — anywhere in the top-5?
  - **judge_yes** — an LLM reads the top-5 and says "could you answer from this?" (mechanics
    deep-dive is Part 4 — here it's just "the robust metric from last time").
- The **new** lesson vs Part 2: `hit@1` and `judge_yes` **diverge** on paraphrase questions —
  search returns the right chunk, the literal expected substring isn't in it → `hit@1 = False`
  but `judge_yes = True`. Show one such row. That's the tease for Part 4.
- pytest siblings that lock it in: `test_retrieval_strategies.py`, `test_rrf_vs_union.py`,
  `test_rerank.py`, `test_experiments_runner.py` (all `phase2`).

### Experiments file for the shoot

```yaml
experiments:
  - {name: vector,       chunking: semantic, search_type: vector,       use_hyde: false, match_threshold: 0.50}
  - {name: text,         chunking: semantic, search_type: text,         use_hyde: false}
  - {name: hybrid_rrf,   chunking: semantic, search_type: hybrid_rrf,   use_hyde: false, match_threshold: 0.50}
  - {name: hybrid_union, chunking: semantic, search_type: hybrid_union, use_hyde: false, match_threshold: 0.50}
```

### Hypothesis to commit on camera (before the reveal)

> - **text** wins the exact-fact `empresa` questions, craters on the `lead`/paraphrase group
>   (often returns nothing).
> - **vector** is the mirror image — fine on paraphrase, wobbly on the pure-number questions.
> - **hybrid_rrf ≥ both** everywhere — that's the point.
> - **hybrid_union < hybrid_rrf** — sorting incomparable scores barely beats using vector alone.

Then run / reveal. Don't fake numbers — this section fills in after a real `cli.py eval --fast`.

**Results (paste after running):**

```
(pending — run: PYTHONPATH=. .venv/bin/python eval/run.py --fast)
```

---

## Production checklist

- [ ] Pre-run: `make rag_phase0 STRATEGY=semantic EMPRESA=empresa_x` (agent/eval need semantic)
- [ ] `.env` DB creds valid (the Supabase project must be un-paused)
- [ ] Swap `tests/rag/experiments.yaml` to the 4-row file above (keep the old one aside)
- [ ] Pre-run `cli.py eval --fast`, commit `eval/analysis/*` for the shoot
- [ ] Notebook (`docs/videos/search.ipynb`, 32 cells: 4 search types + HyDE/rerank add-ons)
      committed **with outputs** as a fallback
- [ ] `COHERE_API_KEY` valid and not rate-limited before recording the rerank cell (a stale trial
      key just prints "(rerank unavailable: ...)" — decide if that's an acceptable take)
- [ ] Excalidraw frames: zoom-in · vector (arrows + angle) · text (inverted index) · the
      combine problem (two score columns + "≠") · RRF (ranks + formula) · union (the red X) ·
      all-the-knobs
