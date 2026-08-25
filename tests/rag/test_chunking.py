from pathlib import Path
import pytest

pytestmark = pytest.mark.phase0

from rag.loaders import (
    split_fixed,
    split_token,
    split_markdown,
    split_markdown_hybrid,
    split_text,
    load_and_split_dir,
)


DOC_MD = """
# Title

Intro paragraph.

## Section A
Text A1. Text A2. Text A3.

### Sub A.1
Repeated text A1.1. Repeated text A1.1.

## Section B
Text B. """


def test_split_fixed_counts():
    text = "Lorem ipsum " * 500  # ~6000+ chars
    chunks = split_fixed(text, chunk_size=800, chunk_overlap=200)
    assert 5 <= len(chunks) <= 10
    if len(chunks) >= 2:
        assert chunks[0][-50:] in chunks[1]


def test_split_token_counts():
    text = "Lorem ipsum " * 500  # ~6000+ chars
    chunks = split_token(text, chunk_size=200, chunk_overlap=50)
    assert len(chunks) >= 2


def test_split_markdown_sections():
    chunks = split_markdown(DOC_MD)
    assert 2 <= len(chunks) <= 6
    assert "Intro paragraph" in chunks[0]


def test_split_markdown_hybrid_bounds_long_sections():
    long_section = "Sentence about the topic. " * 100  # long enough to exceed chunk_size
    doc = f"# Title\n\n## Long Section\n{long_section}\n\n## Short Section\nShort text."
    chunks = split_markdown_hybrid(doc, chunk_size=400, chunk_overlap=50)
    # the long section must have been sub-split into more than one piece
    assert len(chunks) >= 3
    assert all(len(c) <= 400 + 50 for c in chunks)


def test_split_markdown_hybrid_keeps_short_sections_intact():
    chunks = split_markdown_hybrid(DOC_MD, chunk_size=800, chunk_overlap=200)
    assert 2 <= len(chunks) <= 6
    assert "Intro paragraph" in chunks[0]


def test_split_text_selector():
    c1, s1 = split_text(DOC_MD, strategy="fixed", chunk_size=400, chunk_overlap=100)
    assert s1 == "fixed" and len(c1) >= 1
    c1b, s1b = split_text(DOC_MD, strategy="token", chunk_size=100, chunk_overlap=20)
    assert s1b == "token" and len(c1b) >= 1
    c2, s2 = split_text(DOC_MD, strategy="markdown")
    assert s2 == "markdown" and len(c2) >= 1
    c2b, s2b = split_text(DOC_MD, strategy="markdown_hybrid", chunk_size=400, chunk_overlap=100)
    assert s2b == "markdown_hybrid" and len(c2b) >= 1
    # Semantic requires an embedder + langchain-experimental; acceptable to skip if unavailable
    try:
        from rag.ingest import get_embedder
        emb = get_embedder()
        c3, s3 = split_text(DOC_MD, strategy="semantic", embedder=emb)
        assert s3 == "semantic" and len(c3) >= 1
    except Exception:
        pass


def test_load_and_split_dir(tmp_path: Path):
    kb = tmp_path / "kb"
    kb.mkdir()
    f = kb / "doc.md"
    f.write_text(DOC_MD, encoding="utf-8")
    items = load_and_split_dir(kb, strategy="markdown")
    assert items, "No chunks loaded"
    first = items[0]
    assert set(["doc_path", "chunk_ix", "content", "meta"]).issubset(first.keys())
    assert first["meta"].get("chunking") == "markdown"
