import pytest

pytestmark = pytest.mark.phase0


def test_ingest_graph_two_nodes(tmp_path, db_ready, require_embeddings):
    from rag.ingest import graph
    import psycopg

    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "doc.md").write_text("Company Z policy: 30/60 and 12 months.", encoding="utf-8")

    # 1) Staging only
    res1 = graph.invoke({
        "base_dir": str(kb),
        "empresa": "Company Z",
        "skip_chunks": True,
    })
    assert res1.get("staged") is not None

    # 2) Chunking only (skip staging)
    res2 = graph.invoke({
        "empresa": "Company Z",
        "strategy": "markdown",
        "skip_stage": True,
    })
    assert res2.get("chunked") is not None

    with psycopg.connect(db_ready) as conn:
        with conn.cursor() as cur:
            cur.execute("select count(*) from public.kb_docs where lower(empresa)=lower('Company Z')")
            c_docs = cur.fetchone()[0]
            cur.execute("select count(*) from public.kb_chunks where lower(empresa)=lower('Company Z')")
            c_chunks = cur.fetchone()[0]
            assert c_docs >= 1 and c_chunks >= 1
