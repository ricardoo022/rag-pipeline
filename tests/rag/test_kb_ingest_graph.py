from pathlib import Path
import pytest

pytestmark = pytest.mark.phase0


def test_kb_ingest_graph_runs(tmp_path: Path, db_ready, require_embeddings):
    from rag.ingest import graph
    import psycopg

    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "doc.md").write_text("Product A policies with warranty.", encoding="utf-8")

    res = graph.invoke({"base_dir": str(kb), "strategy": "fixed", "empresa": "Test Company"})
    chunked = res.get("chunked")
    assert isinstance(chunked, int) and chunked >= 1

    with psycopg.connect(db_ready) as conn:
        with conn.cursor() as cur:
            cur.execute("select count(*) from public.kb_chunks where empresa=%s", ("Test Company",))
            c = cur.fetchone()[0]
            assert c >= 1
