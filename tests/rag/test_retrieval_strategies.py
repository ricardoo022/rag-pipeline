from pathlib import Path
import pytest

# Phase 2: search strategies and filters
pytestmark = pytest.mark.phase2

def test_retrieval_hybrid_filters_and_relevance(tmp_path: Path, db_ready, require_embeddings):
    from rag.ingest import graph
    from rag.search import kb_search_client

    # Builds a KB for two empresas
    kb = tmp_path / "kb"
    (kb / "empresa_x").mkdir(parents=True)
    (kb / "empresa_y").mkdir(parents=True)

    fx = kb / "empresa_x" / "produto_a.md"
    fx.write_text("""
# Produto A — Políticas (Empresa X)
[KB_TEST_EMPRESA_X] O Produto A possui garantia de 12 meses e permite pagamento em 30/60 dias.
""".strip(), encoding="utf-8")

    fy = kb / "empresa_y" / "produto_a.md"
    fy.write_text("""
# Produto A — Políticas (Empresa Y)
[KB_TEST_EMPRESA_Y] Para Empresa Y, política diferente sem menção à garantia de 12 meses.
""".strip(), encoding="utf-8")

    # Ingest with an empresa filter (unique name to isolate from the shared DB)
    import uuid as _uuid
    empresa_x_name = f"Empresa X {str(_uuid.uuid4())[:8]}"
    empresa_y_name = f"Empresa Y {str(_uuid.uuid4())[:8]}"
    res1 = graph.invoke({"base_dir": str(kb / "empresa_x"), "strategy": "markdown", "empresa": empresa_x_name})
    res2 = graph.invoke({"base_dir": str(kb / "empresa_y"), "strategy": "markdown", "empresa": empresa_y_name})
    assert (res1.get("chunked") or 0) >= 1 and (res2.get("chunked") or 0) >= 1

    query = "garantia de 12 meses"

    # TEXT
    res_text = kb_search_client.invoke({
        "query": query,
        "k": 5,
        "search_type": "text",
        "reranker": "none",
        "empresa": empresa_x_name,
    })
    assert isinstance(res_text, list)
    assert res_text, "No results in TEXT"
    assert all("[kb_test_empresa_x]" in (it["content"] or "").lower() for it in res_text)
    assert all("[kb_test_empresa_y]" not in (it["content"] or "").lower() for it in res_text)

    # VECTOR
    res_vec = kb_search_client.invoke({
        "query": query,
        "k": 5,
        "search_type": "vector",
        "reranker": "none",
        "empresa": empresa_x_name,
    })
    assert isinstance(res_vec, list)
    assert res_vec, "No results in VECTOR"
    assert all("[kb_test_empresa_x]" in (it["content"] or "").lower() for it in res_vec)
    assert all("[kb_test_empresa_y]" not in (it["content"] or "").lower() for it in res_vec)

    # HYBRID
    res_hybrid = kb_search_client.invoke({
        "query": query,
        "k": 5,
        "search_type": "hybrid",
        "reranker": "none",
        "empresa": empresa_x_name,
    })
    assert isinstance(res_hybrid, list)
    assert res_hybrid, "No results in HYBRID"
    assert all("[kb_test_empresa_x]" in (it["content"] or "").lower() for it in res_hybrid)
    assert all("[kb_test_empresa_y]" not in (it["content"] or "").lower() for it in res_hybrid)
