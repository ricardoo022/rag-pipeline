# Part 2 — Chunking Strategies (plan)

Working title: *"Chunking: the decision that sets the ceiling of your RAG"*
Series position: Part 2 of 3+. Part 1 = client management + RAG overview. Part 2 (this) =
chunking only — **search is deliberately cut** to keep the runtime down and the teaching
focused. Search moves to Part 3, evaluators to Part 4.

Scope: three strategies only — **fixed**, **markdown**, **semantic**. `token` and
`markdown_hybrid` are not mentioned.

---

## The idea in one paragraph

A chunk is the unit of retrieval: search returns chunks, the LLM only sees the chunks search
picked. So how you cut the document decides what is *possible* to answer, before any model or
prompt is involved. You cannot pick a chunking strategy by reasoning about it — you look at the
chunks it produces, then you measure retrieval on real questions. This video does exactly that,
on our own pipeline, and lands on a result that is "wrong" in a useful way: the dumbest strategy
wins *on this data*, and that is the point.

## What the viewer should be able to do afterwards

- Explain the "too big vs too small" tension and why "one chunk ≈ one self-contained idea".
- Describe how fixed / markdown / semantic each decide where to cut.
- Open a notebook, run a strategy on their own docs, and read whether the output is sane.
- Run the eval in this repo and read the `hit@1 / hit@5 / judge_yes` table.
- Resist "just use strategy X" — know that it depends on the documents.

## Two example documents, used the whole video

- `kb/empresa_x/products/produto_a.md` — 779 chars / 151 tokens. Small, clean, one `##` per
  topic. Used to teach the concepts (every chunk fits on screen).
- `kb/empresa_x/lead_reports/lead_maria.md` — 2500 chars / 520 tokens. Long, nested `#/##/###`,
  prose + lists. The realistic hard case.

Do not switch documents mid-video.

---

## Segment 1 — Explain chunking + show the strategies (notebook)

**Talk (whiteboard / slide, ~3 min)**
- Chunk = unit of retrieval. Diagram: doc → chunks → embeddings → DB; query → embedding →
  search returns *chunks* → LLM answers from those chunks only.
- The tension:
  - too big → embedding is an average of many topics, weakly similar to any one question; also
    dumps junk into the LLM context.
  - too small → the answer is split across chunks and retrieval grabs only one; a chunk loses
    the context that made it meaningful (`"- 12 months"` of *what*?).
  - target: one chunk ≈ one self-contained idea.
- Three ways to decide *where to cut*:
  - **fixed** — count characters, cut every N (overlap M), never looks at content. Predictable,
    language-agnostic, blind.
  - **markdown** — cut on the headings the author wrote. Assumes document structure = meaning.
  - **semantic** — embed each sentence, cut where consecutive sentences are most dissimilar.
    Adapts to content; costs an API call per doc; not deterministic.

**Notebook (`notebooks/chunking.ipynb`), cells in order:**

1. **Imports + load both docs**, print each with `len(text)` chars and a `tiktoken` token count.
   Show the raw markdown so viewers hold a picture of the two docs.
2. **`show_chunks(chunks)` helper** — prints each chunk numbered, with `chars / tokens`, and a
   visible separator line between chunks. Legibility is the whole point of this cell.
3. **fixed — `split_fixed(text, 800, 200)`**
   - `produto_a` → **1 chunk = the entire file**. Say out loud: chunk_size 800 on a 779-char doc
     does nothing. This is failure mode #1 and it is silent.
   - `lead_maria` → 4 chunks, each ~750 chars, cutting mid-section (chunk 0 ends inside
     "Context and Objectives").
   - Re-run `split_fixed(lead_maria, 300, 50)` → 13 chunks. Point at chunk 1 = bare
     `"## Context and Objectives"` (orphan heading) and a 67-char sentence fragment. Highlight
     the ~50 chars duplicated between chunk N and N+1 → that's overlap, and why it exists.
4. **markdown — `split_markdown(text)`**
   - `produto_a` → 5 chunks, one per `##` section. Clean. But "Notes and Upgrades" is a 60-char
     fragment, and **the heading text is gone** — chunk 2 is `"- Warranty: standard 12 months…"`
     with no "Warranty" (splitter moves headers to metadata, our code keeps only
     `page_content`).
   - `lead_maria` → 12 chunks; the first is `"- Industry: …"` — `# Lead Report — Maria` and
     `## Executive Summary` both dropped. A "what is Maria's budget?" query must now match a
     chunk that never says "Maria".
   - This is failure mode #2. Decide before filming: fix it first, or make "we found this by
     looking" a beat (see Open question).
5. **semantic — `split_semantic(text, embedder)`** (pre-run; needs OpenRouter)
   - `lead_maria` → show it cut *inside* a section, at a topic shift a heading didn't mark.
     That is its entire reason to exist.
   - `time` the cell; say it costs an API call per document and the boundaries move a little
     between runs.
6. **Summary table** — one DataFrame: rows = {fixed, markdown, semantic}, cols = {n_chunks,
   min/mean/max chars, mean tokens}, for each of the two docs. The single recap frame.
7. **The retrieval bridge (no DB yet)** — take the real question
   *"What are the commercial terms and warranty for Product A?"* (`expected_all: ["30/60",
   "12 months"]`). By eye, mark which chunk(s) hold the answer under each strategy:
   - fixed(800): 1 chunk — both facts present, plus 3 irrelevant sections.
   - markdown: `"30/60"` is in the *Commercial Terms* chunk, `"12 months"` is in the *Warranty*
     chunk — **two different chunks**. A top-1 retrieval can only get half.
   - semantic: depends where it cut — show it.
   - Punchline: chunking already decided whether this question is answerable, before search runs.
     Cut to Segment 2.

---

## Segment 2 — Inject the chunks into the database (pipeline)

**Talk + terminal (~2 min)**
- One command stages the files once, then materializes the chunks three times — once per
  strategy — under separate path prefixes so they don't collide:
  ```
  make rag_phase0 STRATEGY="fixed markdown semantic" EMPRESA=empresa_x
  ```
- Narrate the two phases from `rag/ingest.py`: `stage_docs` (files → `kb_docs`, idempotent) →
  `chunk_docs` (chunk + embed + upsert → `kb_chunks`). Mention embeddings are
  `text-embedding-3-small`, 1536-dim, via OpenRouter.
- Pre-run this before recording (embedding calls are slow and cost money). On camera, run it
  live only if it's fast enough; otherwise show it once then jump to the result.

---

## Segment 3 — Explain our database (~2 min)

- Postgres + **pgvector**, hosted on Supabase. Two tables (point at `docs/database.md`):
  - `kb_docs` — raw staged documents. Idempotent on `(source_path, source_hash)`.
  - `kb_chunks` — one row per chunk: `content`, `embedding vector(1536)`, a generated
    `fts` tsvector (ignored this video — no search), and a `chunking` column.
- The `chunking` column is the whole trick: all three strategies live in the same table, and
  the eval filters on it — so we ask the *same questions* against *the same DB*, changing only
  how the documents were cut.
- Show it concretely:
  ```sql
  select chunking, count(*) from kb_chunks where empresa = 'Empresa X' group by 1;
  ```
  Expect roughly: fixed ≈ few chunks (docs < chunk_size), markdown ≈ many, semantic ≈ middle.
- Retrieval itself is SQL functions (`kb_vector_search`, …) — name them, say "next video".

---

## Segment 4 — Run the test

**Terminal**
```
# experiments file for this video: 3 rows, vector-only, no HyDE
cli.py eval --fast          # first 10 questions (all Empresa X), writes eval/analysis/*.{json,csv}
```
- Pre-run before recording (LLM judge = API calls + non-determinism; see
  `docs/phase1_flakiness.md`). Commit the CSV. Walk through the saved file on camera.

The experiments file (`tests/rag/experiments.yaml` for the shoot):
```yaml
experiments:
  - {name: fixed_vec,    chunking: fixed,    search_type: vector, use_hyde: false, match_threshold: 0.50}
  - {name: markdown_vec, chunking: markdown, search_type: vector, use_hyde: false, match_threshold: 0.50}
  - {name: semantic_vec, chunking: semantic, search_type: vector, use_hyde: false, match_threshold: 0.50}
```
Vector-only and everything else held constant, so the **only** variable is chunking.

---

## Segment 5 — Explain what the test measures (~2 min, notebook or slide)

For each experiment × each question, `eval/run.py`:
1. Runs `kb_search_client` with `k=5`, `search_type=vector`, `chunking=<the strategy>`,
   `empresa="Empresa X"`.
2. Scores the retrieved chunks three ways:
   - **hit@1** — do all expected strings appear in the **top-1** chunk?
   - **hit@5** — do they appear anywhere in the **top-5**?
   - **judge_yes** — an LLM (gpt-4.1-mini, OpenRouter) reads the top-5 + the expected answer and
     says yes/no "could you answer from this?" with a one-line reason.
3. Aggregates per experiment → `hit@1`, `hit@5`, `judge_yes` rates + per-question detail rows.

Key point for the audience: **hit@1 is where fragmentation shows up.** A question like
`expected_all: ["30/60", "12 months"]` needs *both* facts in one chunk to pass hit@1 — so a
strategy that splits them across chunks fails hit@1 even if hit@5 is fine.

---

## Segment 6 — State the hypothesis *before* revealing results (~1 min)

On camera, predict and commit:

> On this corpus, **fixed should win**. The documents are smaller than the 800-char chunk size,
> so "fixed" collapses to "one chunk per document" — nothing gets fragmented, every fact for a
> question sits together. **markdown should lose hit@1 on the multi-fact questions**, because it
> splits Commercial Terms and Warranty into different chunks and throws the headings away.
> **semantic should land in the middle** — better than markdown on grouping, worse than fixed on
> completeness, and slower.

Then run / reveal.

---

## Segment 7 — Show the results (~2 min)

- Open `eval/analysis/*_results.csv`. Read the summary table: `hit@1 / hit@5 / judge_yes` per
  strategy.
- Confirm or correct the hypothesis with the actual numbers (must come from a real pre-run —
  do not fake).
- Drill into 2 detail rows:
  - the `["30/60", "12 months"]` question — show markdown's `hit@1 = False`, its `top1_snippet`,
    and the judge's justification.
  - one where semantic's meaning-based cut helped or hurt.
- If markdown's header-drop clearly cost it points, say so plainly — that's the "measure, don't
  assume" thesis proven on our own code.

---

## Segment 8 — What we should do (~1 min close)

- "On *this* data, fixed won. That is not a recommendation — it's a measurement. Our documents
  are one page each; on a 40-page contract, fixed would be the worst and markdown or semantic
  would win. The framework's value is that you just answered this with a number in five minutes
  instead of guessing."
- Concrete guidance to leave them with:
  - Always `show_chunks()` on a few real docs before you ingest anything.
  - Set `chunk_size` relative to your actual document sizes, not a default.
  - Pick the strategy that keeps *a full answer* in one chunk for your question types.
  - Re-measure whenever the document type changes.
- Tease Part 3: chunking is settled — now build the agent and the search on top, and judge the
  answers.

---

## Production checklist

- [ ] Pre-run: `make rag_phase0 STRATEGY="fixed markdown semantic" EMPRESA=empresa_x`
- [ ] Pre-run: `cli.py eval --fast`, commit `eval/analysis/*` for the shoot
- [ ] Notebook committed **with outputs** as a fallback; re-run only fixed/markdown cells live
- [ ] `tests/rag/experiments.yaml` swapped to the 3-row vector-only file (keep the old one aside)
- [ ] Decide the header-drop question (below)

## Open question — the markdown header-drop

`split_markdown` discards heading text (kept only as metadata, which `loaders.py` drops). It
makes markdown look worse than it should.
- **(a)** Fix it first (prepend the `h1 > h2 > h3` path to each chunk) → fair fight, less drama.
- **(b)** Leave it, make "we caught this by reading the chunks" a beat, fix it in Part 3.

Leaning **(b)** — it proves the entire thesis of the series.
