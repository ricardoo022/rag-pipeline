"""Loaders and splitters for the `.md` files under `kb/`.

Five chunking strategies: fixed-size, token-based, markdown-header-aware,
markdown-hybrid (header-aware with a size-bounded recursive sub-split), and
semantic (embedding-guided). Each fails explicitly rather than silently
falling back to another strategy when it can't run (e.g. semantic chunking
with no embedder or package available).
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


def split_token(text: str, chunk_size: int = 800, chunk_overlap: int = 200) -> List[str]:
    """Like split_fixed, but chunk_size/chunk_overlap count tokens (tiktoken
    cl100k_base) rather than characters — sizes chunks against what embedding
    models actually meter/truncate on."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def split_markdown(text: str, *, chunk_size: int = 800, chunk_overlap: int = 200) -> List[str]:
    """Splits on markdown headers only (h1-h3) — no fixed-size fallback."""
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    docs = splitter.split_text(text)
    return [d.page_content for d in docs]


def split_markdown_hybrid(text: str, *, chunk_size: int = 800, chunk_overlap: int = 200) -> List[str]:
    """Splits on markdown headers first, then recursively sub-splits any
    section still over chunk_size — structure-aware like split_markdown, but
    with an actual size bound instead of one chunk per section regardless of
    length."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    sections = header_splitter.split_text(text)
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks: List[str] = []
    for doc in sections:
        content = doc.page_content
        if len(content) > chunk_size:
            chunks.extend(sub_splitter.split_text(content))
        else:
            chunks.append(content)
    return chunks


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
        raise RuntimeError("Chunking strategy not provided (fixed/token/markdown/markdown_hybrid/semantic)")
    if s == "fixed":
        chunks, resolved = split_fixed(text, chunk_size, chunk_overlap), "fixed"
    elif s == "token":
        chunks, resolved = split_token(text, chunk_size, chunk_overlap), "token"
    elif s == "markdown":
        chunks, resolved = split_markdown(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap), "markdown"
    elif s == "markdown_hybrid":
        chunks, resolved = split_markdown_hybrid(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap), "markdown_hybrid"
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
