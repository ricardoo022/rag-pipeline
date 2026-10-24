# The `split_markdown` header-drop problem (fixed)

`split_markdown` used to discard every heading it split on. That was a bug in our usage of
`MarkdownHeaderTextSplitter`, not a property of "splitting by headers" as a technique — a
correctly-built markdown splitter keeps the heading text attached to its body. **Fixed** in
`rag/loaders.py` (`split_markdown` now prepends an `h1 > h2 > h3` breadcrumb to each chunk). This
doc keeps the original bug writeup for context, plus what the fix actually does.

Note: `split_markdown_hybrid` — the strategy that also applied this fix — was later removed
entirely (along with `split_token`), so this doc now only covers `split_markdown`.

## The bug

```python
def split_markdown(text: str, *, chunk_size: int = 800, chunk_overlap: int = 200) -> List[str]:
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    docs = splitter.split_text(text)
    return [d.page_content for d in docs]   # <- half of what the splitter returned
```

`MarkdownHeaderTextSplitter` doesn't destroy the heading — it deliberately *separates* it: the
heading text becomes `Document.metadata` (e.g. `{"h2": "Warranty and Support"}`), and everything
below it becomes `Document.page_content`. That split is a reasonable library design — it lets the
caller decide whether to use the heading for filtering, display, or reconstruct it into the text.
Our function reads only `page_content` and drops the `Document` object (metadata included)
immediately after, so the heading is gone by the time a chunk reaches embedding or storage.

## Confirmed effect (real output, `docs/videos/chunking.ipynb`, before the fix)

`split_markdown(lead_maria.md)`, chunk 0:

```
- Industry: agricultural-input retail (mid-size)
- Estimated budget: R$ 80,000
- Desired timeline: 60 days
- Focus: standardize the support journey and consolidate information in a single dashboard
```

Both `"# Lead Report — Maria (Empresa X)"` and `"## Executive Summary"` are gone. Nothing in this
chunk's text says "Maria" — a question about Maria's budget now has to match a chunk that never
mentions her.

Same pattern on `produto_a.md`: the *Warranty and Support* chunk becomes
`"- Warranty: standard 12 months for manufacturing defects. - Support: ..."` — with no
"Warranty" anywhere in it.

## Why this matters (not just a cosmetic gap)

1. **Vector search** — the embedding is computed from `content` alone. A chunk missing its own
   topic word is weakly similar to a query that uses that word ("what's the warranty on Product
   A?" embeds "warranty" strongly; the chunk that should answer it doesn't).
2. **Full-text search** — `kb_chunks.fts` is `to_tsvector('english', content)` (`sql/01_init.sql`).
   No heading in `content` means no lexical match on the heading's words either.
3. **The agent's context** — `agent/rag_agent.py`'s prompt asks for grounded answers with
   `doc_path#chunk_ix` citations. A chunk with no indication of what section it came from is
   harder to justify a citation for, and easier to misattribute if two documents share the same
   bullet-point shape.

In short: the heading is usually the single most information-dense phrase in the section — often
literally the keyword a real question will use. Throwing it away reproduces the "chunk too small,
loses its own context" failure mode from a strategy that was supposed to avoid it by cutting on
meaningful boundaries.

## What a correctly-built markdown splitter does instead

1. Builds a header tree — each piece of text knows its `h1 > h2 > h3` path (this is exactly what
   `MarkdownHeaderTextSplitter`'s `metadata` already gives you, for free).
2. **Reconstructs a breadcrumb into the chunk's own content**, not just metadata — because
   embedding only ever sees `content`. Standard practice (LlamaIndex's `MarkdownNodeParser`, the
   `unstructured` library, and LangChain's own RAG examples all do this): something like
   ```
   Product A > Warranty and Support

   - Warranty: standard 12 months for manufacturing defects.
   ```
   instead of the body alone. The chunk becomes self-contained — it reads sensibly on its own,
   out of order, without the rest of the document nearby.
3. Keeps the metadata too, separately, for anyone who wants to filter/facet by section without
   parsing the reconstructed text.
4. Never cuts inside a list or table — treats them as atomic blocks even if that means exceeding
   `chunk_size`, rather than letting a generic recursive splitter slice through the middle of a
   bullet.
5. Merges trivial/empty sections (a `##` with no text under it, only subheadings) instead of
   emitting a content-free metadata-only chunk.

## The fix (applied)

`rag/loaders.py` now has two small helpers used by both splitters:

```python
def _header_breadcrumb(metadata: Dict[str, Any]) -> str:
    return " > ".join(metadata[k] for k in ("h1", "h2", "h3") if k in metadata)

def _with_breadcrumb(content: str, breadcrumb: str) -> str:
    return f"{breadcrumb}\n\n{content}" if breadcrumb else content
```

`split_markdown` prepends the breadcrumb to every chunk.

A few lines, not a rewrite. Covered by `tests/rag/test_chunking.py`.

## Real output after the fix

`split_markdown(produto_a.md)`, the *Warranty and Support* chunk:

```
Product A — Specifications and Policies (Empresa X) > Warranty and Support

- Warranty: standard 12 months for manufacturing defects.
- Support: standard channel with response times during business hours.
```

`split_markdown(lead_maria.md)`, chunk 0:

```
Lead Report — Maria (Empresa X) > Executive Summary

- Industry: agricultural-input retail (mid-size)
- Estimated budget: R$ 80,000
- Desired timeline: 60 days
- Focus: standardize the support journey and consolidate information in a single dashboard
```

## Status

Fixed directly in `rag/loaders.py`, ahead of the original plan. `docs/videos/part2-chunking-strategies.md`'s
"Open question" had leaned toward keeping the bug visible on camera as a "we caught this by
reading the chunks" beat (option b); that plan is now stale on this point since the fix landed
before recording — the video script needs a pass to either show the *before* state from this doc
as a re-enacted example, or drop that beat and demonstrate the breadcrumb directly.
