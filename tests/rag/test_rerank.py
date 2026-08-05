from pathlib import Path
import pytest

# Phase 2: optional reranker (integration)
pytestmark = pytest.mark.phase2

def test_rerank_with_cohere_smoke(tmp_path: Path, db_ready, require_embeddings, require_cohere):
    """Smoke test: ensures the reranker=cohere path runs and returns a list.

    Minimal dataset; the goal is integration validation, not a strong metric.
    """
    from rag.ingest import graph
    from rag.search import kb_search_client

    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "doc1.md").write_text("Garantia de 12 meses para Produto A.", encoding="utf-8")
    (kb / "doc2.md").write_text("Texto genérico sem relação.", encoding="utf-8")

    graph.invoke({"base_dir": str(kb), "strategy": "fixed", "empresa": "Empresa X"})

    try:
        res = kb_search_client.invoke({
            "query": "garantia de 12 meses",
            "k": 3,
            "search_type": "hybrid",
            "reranker": "cohere",
            "rerank_candidates": 8,
            "empresa": "Empresa X",
        })
    except Exception as e:
        try:
            from cohere.errors import TooManyRequestsError
            if isinstance(e, TooManyRequestsError) or "TooManyRequests" in str(e):
                pytest.skip("Cohere rate-limited (429) — skipping rerank smoke test")
        except Exception:
            pass
        raise
    assert isinstance(res, list)
    assert len(res) >= 1
