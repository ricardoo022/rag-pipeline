# Part 2 — proving it: `test_phase1_chunking_vector_only.py`

Companion to `part2-chunking-strategies.md` and `docs/videos/chunking.ipynb`. The notebook builds
intuition by eye ("this chunk looks like it lost its context"); this test is the segment that
actually measures it against real vector search, so the video's central claim ("chunking decides
what's answerable, before search even runs") is proven, not just asserted.

**File:** `tests/rag/test_phase1_chunking_vector_only.py`
**Run:** `.venv/bin/python -m pytest tests/rag/test_phase1_chunking_vector_only.py -v -s`
(`-s` is required — without it pytest swallows the printed tables.)

Deliberately separate from the notebook: this test needs a real DB connection and makes real
embedding calls (isolated under a throwaway `empresa` UUID, so it doesn't touch `kb/empresa_x`),
and uses its own 8 synthetic documents / 13 questions rather than `produto_a.md`/`lead_maria.md`
— it needs enough questions for a hit-rate number to mean something, which one document can't
give it. Mixing this into the notebook would make a "no DB needed" artifact suddenly require one.

---

## What to say, walking through the code

### 1. Why this test exists (say this before showing any code)

"Everything so far was us looking at chunks and guessing what would go wrong. That's a hypothesis,
not a result. This test runs the actual question through actual vector search, for each strategy,
and counts how often the right answer comes back — no guessing, no LLM opinion, just: is the
expected keyword in the chunk that search returned, yes or no."

### 2. The synthetic KB (lines ~18-236)

Eight short documents, each stating one clear fact (warranty, payment terms, SLA, cancellation
policy, etc.), written **three times** — once per `x_fixed`/`x_markdown`/`x_semantic` directory —
plus three more documents under `y_*` for a second, unrelated "Empresa Y".

**Say:** "Each doc is one paragraph, one fact — this isn't about complex documents, it's about
isolating chunking as the only variable. Empresa Y exists purely as noise: it proves the search
filter by `empresa` actually works, so a hit isn't an accident of a small dataset."

### 3. Staging once, materializing three times (lines ~242-251)

```python
stage_docs_from_dir(str(kb / "x_markdown"), empresa=empresa)
...
materialize_chunks_from_staging(strategy="fixed", empresa=empresa, doc_path_prefix="fixed")
materialize_chunks_from_staging(strategy="markdown", empresa=empresa, doc_path_prefix="markdown")
materialize_chunks_from_staging(strategy="semantic", empresa=empresa, doc_path_prefix="semantic")
```

**Say:** "This mirrors the real pipeline: stage the raw text once, then chunk it three different
ways into the same database, tagged by `doc_path_prefix` so they don't collide. Same source text,
same DB, only the chunking strategy changes between the three — that's what makes it a fair
comparison instead of three different experiments."

### 4. The 13 questions (lines ~254-268)

Each is a real question with a known, short expected keyword — e.g. `("What is the warranty for
Product A?", "12-month")`. No ambiguity about what counts as a correct answer.

**Say:** "These aren't trick questions — one clear fact per document, one clear keyword we expect
back. That's on purpose: if a strategy fails here, on the easiest possible case, that's a real
signal, not noise."

### 5. Running the same question against all three strategies (lines ~279-308)

```python
res = kb_search_client.invoke({
    "query": q, "k": 5, "search_type": "vector", "reranker": "none",
    "empresa": empresa_lookup, "chunking": chunking,
})
got5 = any(norm in (it["content"] or "").lower() for it in res)
got1 = norm in (res[0]["content"] or "").lower() if res else False
```

**Say:** "Same query, same k=5, same plain vector search — `reranker=none`, no HyDE — the only
thing that changes between runs is `chunking`. `hit@5`: did the expected word show up anywhere in
the top 5 results? `hit@1`: did it show up in the single best result? That second one is the
tougher, more realistic bar — it's what matters if your agent only reads the top hit."

### 6. What gets printed

Two tables: a summary (`hit@5`/`hit@1` per strategy, averaged over all 13 questions), and a
per-question breakdown showing exactly which document/chunk each strategy pulled back for each
question — so a failure isn't just a percentage, you can see *which* question broke and *why*.

**Say:** "This second table is the important one for the discussion — it's not enough to know
markdown scored lower, we want to see the specific question where it failed and read the chunk it
actually returned."

---

## Then: show the results

Run the command above, paste the two real tables here, and discuss them against the hypothesis
from the notebook (does `markdown` underperform on questions where the fact sits right at a
heading boundary? does `semantic`'s non-determinism show up as a different result on a re-run?).
Do not fabricate numbers — this section only gets filled in after an actual run.

**Results (paste after running):**

```
(pending — run: .venv/bin/python -m pytest tests/rag/test_phase1_chunking_vector_only.py -v -s)
```
