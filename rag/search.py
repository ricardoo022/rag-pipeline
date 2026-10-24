"""Retrieval: `kb_search_client`, a parametrized search tool with optional HyDE
query expansion and optional reranking (e.g. Cohere).

Filter policy: client_id takes priority over empresa; querying with neither (and no
`chunking` filter either, which is allowed for test/benchmark scenarios) is rejected.
"""

import os
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from rag.db import get_conn
from rag.ingest import get_embedder, vec_to_literal

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def query_candidates(
    query: str,
    k: int,
    search_type: str,
    client_id: Optional[str],
    empresa: Optional[str],
    chunking: Optional[str],
    query_embedding: Optional[List[float]],
    match_threshold: Optional[float],
) -> List[Dict[str, Any]]:
    """Fetches candidates from Postgres per `search_type`.

    Returns dicts with doc_path, chunk_ix, content, score, meta.
    """
    params: Dict[str, Any] = {
        "k": k,
        "client_id": client_id,
        "empresa": empresa,
        "chunking": chunking,
        "query": query,
        "threshold": match_threshold,
    }
    with get_conn() as conn:
        with conn.cursor() as cur:
            results: List[Dict[str, Any]] = []
            if search_type in ("vector",):
                if not query_embedding:
                    raise RuntimeError("search_type=vector requires a query embedding")
                vec_lit = vec_to_literal(query_embedding)
                cur.execute(
                    "select doc_path, chunk_ix, content, score, meta from public.kb_vector_search(%(vec)s, %(k)s, %(threshold)s, %(client_id)s, %(empresa)s, %(chunking)s)",
                    {**params, "vec": vec_lit},
                )
            elif search_type in ("text",):
                cur.execute(
                    "select doc_path, chunk_ix, content, score, meta from public.kb_text_search(%(query)s, %(k)s, %(client_id)s, %(empresa)s, %(chunking)s)",
                    params,
                )
            elif search_type in ("hybrid", "hybrid_rrf"):
                if query_embedding is None:
                    raise RuntimeError("search_type=hybrid requires a query embedding")
                vec_lit = vec_to_literal(query_embedding)
                cur.execute(
                    "select doc_path, chunk_ix, content, score, meta from public.kb_hybrid_search(%(query)s, %(vec)s, %(k)s, %(threshold)s, %(client_id)s, %(empresa)s, %(chunking)s)",
                    {**params, "vec": vec_lit},
                )
            elif search_type in ("hybrid_union",):
                if query_embedding is None:
                    raise RuntimeError("search_type=hybrid_union requires a query embedding")
                vec_lit = vec_to_literal(query_embedding)
                cur.execute(
                    "select doc_path, chunk_ix, content, score, meta from public.kb_hybrid_union(%(query)s, %(vec)s, %(k)s, %(threshold)s, %(client_id)s, %(empresa)s, %(chunking)s)",
                    {**params, "vec": vec_lit},
                )
            else:
                raise RuntimeError(f"Unknown search_type: {search_type}")
            for row in cur.fetchall() or []:
                results.append({
                    "doc_path": row[0],
                    "chunk_ix": row[1],
                    "content": row[2],
                    "score": float(row[3]) if row[3] is not None else None,
                    "meta": row[4] or {},
                })
            return results


def _cohere_with_retry(call):
    # Trial keys allow 10 calls/min; back off instead of failing. COHERE_MAX_RETRIES=0 disables.
    import time

    from cohere.errors import TooManyRequestsError

    retries = int(os.getenv("COHERE_MAX_RETRIES", "6"))
    for attempt in range(retries + 1):
        try:
            return call()
        # ValueError covers occasional garbled 200 bodies (UnicodeDecodeError/JSONDecodeError).
        except (TooManyRequestsError, ValueError):
            if attempt == retries:
                raise
            time.sleep(min(10 * (attempt + 1), 60))


def apply_rerank(query: str, items: List[Dict[str, Any]], reranker: str, k: int) -> List[Dict[str, Any]]:
    if reranker == "none" or not items:
        return items[:k]
    if reranker == "cohere":
        try:
            from langchain_cohere import CohereRerank
        except Exception as e:
            raise RuntimeError("reranker='cohere' requested but langchain-cohere is not available") from e
        if not os.getenv("COHERE_API_KEY"):
            raise RuntimeError("reranker='cohere' requested but COHERE_API_KEY is not set")
        reranker_model = CohereRerank(model="rerank-english-v3.0")
        docs = [it["content"] for it in items]
        # Tries multiple APIs for cross-version compatibility.
        results = None
        if hasattr(reranker_model, "rank"):
            results = reranker_model.rank(query=query, documents=docs)
        elif hasattr(reranker_model, "rerank"):
            results = _cohere_with_retry(lambda: reranker_model.rerank(query=query, documents=docs))
        else:
            try:
                results = reranker_model.invoke({"query": query, "documents": docs})
            except Exception:
                results = None

        if isinstance(results, list) and results:
            def _get_content_and_score(obj):
                c = getattr(obj, "document", None)
                if c is not None:
                    txt = getattr(c, "page_content", None) or getattr(c, "text", None) or str(c)
                else:
                    txt = getattr(obj, "page_content", None) or getattr(obj, "text", None)
                if txt is None and isinstance(obj, dict):
                    d = obj.get("document") or obj
                    txt = d.get("text") or d.get("page_content") or str(d)
                score = getattr(obj, "relevance_score", None)
                if score is None and isinstance(obj, dict):
                    score = obj.get("relevance_score")
                return (str(txt or ""), float(score or 0.0))

            res_pairs = [_get_content_and_score(r) for r in results]
            # Maps doc -> queue of indices (supports possible duplicates).
            from collections import defaultdict, deque
            idx_map = defaultdict(deque)
            for i, d in enumerate(docs):
                idx_map[d].append(i)
            ordered = []
            for content, _score in res_pairs:
                if content in idx_map and idx_map[content]:
                    i = idx_map[content].popleft()
                    ordered.append(items[i])
            if ordered:
                return ordered[:k]

        # Fallback: couldn't reorder, return the original top-k.
        return items[:k]
    return items[:k]


def hyde(query: str) -> str:
    """Generates a hypothetical relevant document (HyDE) to expand the query."""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY (required for HyDE via OpenRouter)")
    llm = ChatOpenAI(
        model=os.getenv("HYDE_LLM_MODEL", "openai/gpt-4.1-mini"),
        temperature=0.3,
        openai_api_base=OPENROUTER_BASE_URL,
        openai_api_key=api_key,
    )
    prompt = (
        "Write a concise paragraph that would be highly relevant to the following question, "
        "simulating a real technical document.\nQuestion: " + query
    )
    return (llm.invoke(prompt).content or "").strip()


@tool
def kb_search_client(
    query: str,
    k: int = 5,
    search_type: str = "hybrid",
    reranker: str = "none",
    rerank_candidates: int = 24,
    client_id: Optional[str] = None,
    empresa: Optional[str] = None,
    chunking: Optional[str] = None,
    tags: Optional[List[str]] = None,
    use_hyde: bool = False,
    match_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Searches the KB with filters and optional reranking.

    Filter policy:
    - Prefer client_id when available, otherwise empresa.
    - For test/benchmark scenarios, a global query is allowed when filtering by 'chunking'.
    """
    if not client_id and not empresa and not chunking:
        raise RuntimeError("Querying the KB without a filter (client_id/empresa) or 'chunking' is not allowed")

    query_emb: Optional[List[float]] = None
    if search_type != "text":
        q_for_embed = hyde(query) if use_hyde else query
        query_emb = get_embedder().embed_query(q_for_embed)

    candidates = query_candidates(
        query,
        rerank_candidates or k,
        search_type,
        client_id,
        empresa,
        chunking,
        query_emb,
        match_threshold,
    )

    return apply_rerank(query, candidates, reranker, k)
