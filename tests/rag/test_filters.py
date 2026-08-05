import pytest

# Phase 0: filter policy
pytestmark = pytest.mark.phase0
from unittest.mock import patch


def test_filter_priority_user_then_company(monkeypatch):
    """Exercises the filter priority: client_id -> empresa -> no result.

    Note: this only tests the call skeleton; inserting KB data and associating
    client_id/empresa should be done in dedicated integration tests once the
    environment is provisioned.
    """
    from rag.search import kb_search_client

    # (c) No filters -> logical error
    with pytest.raises(RuntimeError):
        kb_search_client.invoke({
            "query": "Termos do Produto A",
            "k": 3,
            "search_type": "hybrid",
            "reranker": "none",
            # no client_id and no empresa
        })

    # For (b) and (a) we avoid DB/API dependencies by monkeypatching the candidates
    with patch("rag.search.query_candidates", return_value=[]):
        # (b) empresa only -> allowed (not rejected by policy)
        try:
            kb_search_client.invoke({
                "query": "Termos do Produto A",
                "k": 3,
                "search_type": "text",
                "reranker": "none",
                "empresa": "Empresa X",
            })
        except RuntimeError:
            pytest.fail("Query filtered by empresa should not fail policy check")

        # (a) client_id -> allowed (preferred)
        try:
            kb_search_client.invoke({
                "query": "Termos do Produto A",
                "k": 3,
                "search_type": "vector",
                "reranker": "none",
                "client_id": "00000000-0000-0000-0000-000000000000",
            })
        except RuntimeError:
            pytest.fail("Query filtered by client_id should not fail policy check")
