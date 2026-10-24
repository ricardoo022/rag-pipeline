"""Loaders and splitters for the `.md` files under `kb/`.

Three chunking strategies: fixed-size, markdown-header-aware, and semantic
(embedding-guided). Each fails explicitly rather than silently falling back
to another strategy when it can't run (e.g. semantic chunking with no
embedder or package available).
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

_HAS_SEMANTIC = False
SemanticChunker = None  # type: ignore
try:  # some langchain-text-splitters versions expose it here
    from langchain_text_splitters import SemanticChunker as _SC  # type: ignore
    SemanticChunker = _SC  # type: ignore
    _HAS_SEMANTIC = True
except Exception:
    try:  # newer langchain-experimental versions
        from langchain_experimental.text_splitter import SemanticChunker as _SC  # type: ignore
        SemanticChunker = _SC  # type: ignore
        _HAS_SEMANTIC = True
    except Exception:
        SemanticChunker = None  # type: ignore
        _HAS_SEMANTIC = False


def list_md_files(base_dir: str | Path) -> List[Path]:
    base = Path(base_dir)
    return sorted([p for p in base.rglob("*.md") if p.is_file()])


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def split_fixed(text: str, chunk_size: int = 800, chunk_overlap: int = 200) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)


_HEADER_KEYS = ("h1", "h2", "h3")


def _header_breadcrumb(metadata: Dict[str, Any]) -> str:
    """Joins whatever h1/h2/h3 levels MarkdownHeaderTextSplitter captured into
    a single "h1 > h2 > h3" line, e.g. "Product A > Warranty and Support"."""
    return " > ".join(metadata[k] for k in _HEADER_KEYS if k in metadata)


def _with_breadcrumb(content: str, breadcrumb: str) -> str:
    return f"{breadcrumb}\n\n{content}" if breadcrumb else content


def split_markdown(text: str, *, chunk_size: int = 800, chunk_overlap: int = 200) -> List[str]:
    """Splits on markdown headers only (h1-h3) — no fixed-size fallback.

    MarkdownHeaderTextSplitter moves the heading text itself into
    `Document.metadata`, leaving `page_content` as just the body — so each
    chunk is prefixed here with its "h1 > h2 > h3" breadcrumb before the
    metadata (and the heading words in it) would otherwise be lost. Without
    this, a chunk like the body of "## Warranty and Support" never contains
    the word "Warranty", which starves both the embedding and `fts` of the
    section's own topic word. See docs/markdown_header_drop.md.
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    docs = splitter.split_text(text)
    return [_with_breadcrumb(d.page_content, _header_breadcrumb(d.metadata)) for d in docs]


def split_semantic(text: str, embedder=None) -> List[str]:
    if not _HAS_SEMANTIC:
        raise RuntimeError("SemanticChunker is not available in this environment (langchain-experimental)")
    if embedder is None:
        raise RuntimeError("Semantic chunking requires a valid 'embedder' (e.g. OpenAIEmbeddings)")
    splitter = SemanticChunker(embedder, breakpoint_threshold_type="interquartile")
    return splitter.split_text(text)


def split_text(
    text: str,
    strategy: str = "fixed",
    *,
    embedder=None,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> Tuple[List[str], str]:
    """Splits text per the given strategy. Returns (chunks, resolved_strategy)."""
    s = (strategy or "").lower()
    if not s:
        raise RuntimeError("Chunking strategy not provided (fixed/markdown/semantic)")
    if s == "fixed":
        chunks, resolved = split_fixed(text, chunk_size, chunk_overlap), "fixed"
    elif s == "markdown":
        chunks, resolved = split_markdown(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap), "markdown"
    elif s == "semantic":
        chunks, resolved = split_semantic(text, embedder), "semantic"
    else:
        raise RuntimeError(f"Invalid chunking strategy: {strategy}")
    # Splitters (SemanticChunker in particular) can emit trailing blank chunks —
    # useless for retrieval and rejected as embedding input by stricter providers.
    return [c for c in chunks if c.strip()], resolved


def load_and_split_dir(
    base_dir: str | Path,
    strategy: str = "fixed",
    *,
    embedder=None,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> List[Dict[str, Any]]:
    """Loads every `.md` under base_dir and returns chunks as
    [{doc_path, chunk_ix, content, meta: {chunking}}]."""
    items: List[Dict[str, Any]] = []
    for path in list_md_files(base_dir):
        text = read_text(path)
        chunks, resolved = split_text(
            text, strategy, embedder=embedder, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        for i, c in enumerate(chunks):
            items.append({
                "doc_path": str(path.as_posix()),
                "chunk_ix": i,
                "content": c,
                "meta": {"chunking": resolved},
            })
    return items
