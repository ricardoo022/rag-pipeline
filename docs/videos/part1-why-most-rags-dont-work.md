# Part 1 — Why Most RAGs Don't Actually Work

- **URL:** https://youtu.be/ednN-O0qqfU
- **Title:** "Why Most RAG'S Don't Actually Work: PART 1"
- **Role:** first video of a series that uses this repo as a reusable RAG framework. Deliberately
  a blueprint/overview — "you have AI to answer the technical questions; the point is knowing
  what to ask." Little code on screen.

## Core argument

Most RAGs (including ones the author built in n8n and in code) fail because you can't say *why*
they work or don't:

- No way to test and measure → no data to choose a **chunking strategy**, a **search strategy**,
  or a **model**. "If we don't know why we're winning, we can't win again; if we don't know why
  we're failing, we can't fix it."
- Two recurring failure modes:
  1. You evaluate the agent by asking it questions *yourself*.
  2. You have no data to "sleep well at night" — so you reach for bigger models and burn money
     on faith.
- Goal of the series: build a RAG where **data drives the decisions**.

## RAG overview presented

Ingestion: client PDFs → convert to `.md` → strategic chunking (the on-screen example is
markdown-header chunking: split on titles, keep each title with its section) → embedding model
turns text into vectors → store in the vector DB.

Query: user question → same embedding model → question vector → search the vector DB (several
search strategies exist; the series will test more than one).

Maps to this repo: `rag/loaders.py` (chunking), `rag/ingest.py` (stage → chunk → embed),
`sql/` (`kb_docs`/`kb_chunks` + retrieval functions), `rag/search.py` (query side).

## The client-management focus (main content of Part 1)

How to run a RAG project for a client. You need more than their documents:

- Ask for the documents **and** ~3 hours of the client's time.
- The client produces evaluation questions, each with:
  - the specific information that **must** appear in a correct answer;
  - how the client themselves would phrase the answer (different people answer differently).
- On-screen example: *"What are the commercial terms and warranty for Product A?"* →
  *"30 / 60 / 12 months"*.

This is exactly the shape of `tests/rag/rag_queries.yaml` (`pergunta` / `expected` /
`answer`), and that example lines up with `kb/empresa_x`.

## Series roadmap (as stated on camera)

- **Part 1** — project/client management + RAG overview. *(this video)*
- **Part 2** — chunking strategies + search strategies, tested with measures to pick the best.
  → repo Phase 1/2, `eval/run.py`.
- **Part 3** — evaluators: LLM-as-judge on the agent's final answers, again to get data on why
  it wins or fails. → repo Phase 3, `eval/run_agent.py`.
