"""Two-phase ingestion: stage raw `.md` files into `kb_docs`, then materialize
chunks (+ embeddings) into `kb_chunks` per chunking strategy.

Also exposes a compiled LangGraph graph (`graph`) with two nodes — stage_docs
then chunk_docs — so the same pipeline can be driven from LangGraph Studio or
any other LangGraph-aware caller, not just the CLI.
"""

import hashlib
import mimetypes
import os
from json import dumps as json_dumps
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langchain_openai import OpenAIEmbeddings
from langgraph.graph import StateGraph

from rag.db import get_conn
from rag.loaders import load_and_split_dir, split_text

# Routed through OpenRouter's OpenAI-compatible /embeddings endpoint rather than
# OpenAI directly. Model id keeps the "openai/" provider prefix OpenRouter requires,
# and still resolves to the same 1536-dim vectors kb_chunks.embedding is built for.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "openai/text-embedding-3-small"


def vec_to_literal(v: List[float]) -> str:
    """Formats a float vector as the literal pgvector expects: "[v1,v2,...]"."""
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def get_embedder() -> OpenAIEmbeddings:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY (required for embeddings via OpenRouter)")
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_base=OPENROUTER_BASE_URL, openai_api_key=api_key)


def embed_texts(texts: List[str]) -> List[List[float]]:
    return get_embedder().embed_documents(texts)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def stage_docs_from_dir(base_dir: str = "kb", *, empresa: Optional[str] = None, client_id: Optional[str] = None) -> int:
    """Reads every `.md` under base_dir and inserts it into kb_docs (staging).

    Idempotent via (source_path, source_hash) — an unchanged file is a no-op.
    """
    files = sorted([p for p in Path(base_dir).rglob("*.md") if p.is_file()])
    if not files:
        return 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            count = 0
            for path in files:
                text = path.read_text(encoding="utf-8")
                mime = mimetypes.guess_type(str(path))[0] or "text/markdown"
                cur.execute(
                    """
                    insert into public.kb_docs (source_path, source_hash, mime_type, content, meta, client_id, empresa)
                    values (%s, %s, %s, %s, %s::jsonb, %s::uuid, %s)
                    on conflict (source_path, source_hash) do nothing
                    """,
                    (str(path.as_posix()), sha256_text(text), mime, text, json_dumps({}), client_id, empresa),
                    prepare=False,
                )
                count += 1
    return count


def upsert_chunks(rows: List[Dict[str, Any]], *, client_id: Optional[str] = None, empresa: Optional[str] = None) -> int:
    """Upserts chunks idempotently by (doc_path, chunk_ix). Returns rows written."""
    if not rows:
        return 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            count = 0
            for r in rows:
                cur.execute(
                    """
                    insert into public.kb_chunks (doc_path, chunk_ix, content, embedding, meta, client_id, empresa)
                    values (%s, %s, %s, %s::vector(1536), %s::jsonb, %s::uuid, %s)
                    on conflict (doc_path, chunk_ix)
                    do update set content=excluded.content, embedding=excluded.embedding, meta=excluded.meta,
                                  client_id=excluded.client_id, empresa=excluded.empresa, updated_at=now()
                    """,
                    (
                        r["doc_path"],
                        r["chunk_ix"],
                        r["content"],
                        vec_to_literal(r["embedding"]),
                        json_dumps(r.get("meta") or {}),
                        client_id,
                        empresa,
                    ),
                    prepare=False,
                )
                count += 1
    return count


def materialize_chunks_from_staging(
    *,
    strategy: str = "fixed",
    empresa: Optional[str] = None,
    client_id: Optional[str] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
    path_prefix: Optional[str] = None,
) -> int:
    """Reads kb_docs (optionally filtered by empresa/client_id/path_prefix), chunks
    each document per `strategy`, embeds, and upserts into kb_chunks.

    Re-chunking with a different strategy doesn't require re-reading source files —
    it only reads from kb_docs.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            clauses = []
            vals: List[Any] = []
            if empresa:
                clauses.append("lower(empresa) = lower(%s)")
                vals.append(empresa)
            if client_id:
                clauses.append("client_id = %s::uuid")
                vals.append(client_id)
            if path_prefix:
                like = path_prefix if path_prefix.endswith("%") else f"{path_prefix.rstrip('/')}/%"
                clauses.append("source_path like %s")
                vals.append(like)
            where = (" where " + " and ".join(clauses)) if clauses else ""
            cur.execute(
                f"select source_path, content from public.kb_docs{where}",
                tuple(vals),
                prepare=False,
            )
            docs = cur.fetchall() or []
    if not docs:
        return 0

    embedder = get_embedder() if strategy == "semantic" else None
    rows: List[Dict[str, Any]] = []
    for source_path, content in docs:
        chunks, resolved = split_text(content, strategy=strategy, embedder=embedder, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, c in enumerate(chunks):
            rows.append({
                "doc_path": source_path,
                "chunk_ix": i,
                "content": c,
                "meta": {"chunking": resolved},
            })

    vectors = embed_texts([r["content"] for r in rows])
    for r, vec in zip(rows, vectors):
        r["embedding"] = vec

    return upsert_chunks(rows, client_id=client_id, empresa=empresa)


def truncate_kb_tables() -> None:
    """Truncates kb_chunks and kb_docs. Destructive — use for a clean re-run only."""
    with get_conn() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("truncate table public.kb_chunks;")
            cur.execute("truncate table public.kb_docs;")


# ---------- LangGraph pipeline ----------

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


def node_stage(state: IngestState) -> IngestState:
    if state.get("skip_stage"):
        return {"staged": 0}
    n = stage_docs_from_dir(
        state.get("base_dir") or "kb",
        empresa=state.get("empresa"),
        client_id=state.get("client_id"),
    )
    return {"staged": n}


def node_chunk(state: IngestState) -> IngestState:
    if state.get("skip_chunks"):
        return {"chunked": 0}
    kwargs: Dict[str, Any] = {
        "strategy": state.get("strategy") or "fixed",
        "empresa": state.get("empresa"),
        "client_id": state.get("client_id"),
    }
    if isinstance(state.get("chunk_size"), int):
        kwargs["chunk_size"] = state["chunk_size"]
    if isinstance(state.get("chunk_overlap"), int):
        kwargs["chunk_overlap"] = state["chunk_overlap"]
    n = materialize_chunks_from_staging(**kwargs)
    return {"chunked": n}


def compile_graph():
    g = StateGraph(IngestState)
    g.add_node("stage_docs", node_stage)
    g.add_node("chunk_docs", node_chunk)
    g.set_entry_point("stage_docs")
    g.add_edge("stage_docs", "chunk_docs")
    g.set_finish_point("chunk_docs")
    return g.compile()


graph = compile_graph()
