from pathlib import Path
import pytest

pytestmark = pytest.mark.phase0


def test_kb_ingest_smoke(tmp_path: Path, db_ready, require_embeddings):
    from rag.ingest import graph
    import psycopg

    kb = tmp_path / "kb"
    kb.mkdir()
    f = kb / "produto_a.md"
    f.write_text("""
# Product A — Policies
30/60 day payment terms. Standard 12-month warranty.
""".strip(), encoding="utf-8")

    # Ingestion 1 (stage + chunk in one graph run)
    res1 = graph.invoke({"base_dir": str(kb), "strategy": "markdown"})
    assert (res1.get("chunked") or 0) >= 1

    with psycopg.connect(db_ready) as conn:
        with conn.cursor() as cur:
            cur.execute("select count(*) from public.kb_chunks where doc_path like %s", (str(f.as_posix()),))
            c1 = cur.fetchone()[0]
            assert c1 >= 1

    # Ingestion 2 (idempotent — same content, same doc_path/chunk_ix, upserts in place)
    res2 = graph.invoke({"base_dir": str(kb), "strategy": "markdown"})
    assert (res2.get("chunked") or 0) >= 1
    with psycopg.connect(db_ready) as conn:
        with conn.cursor() as cur:
            cur.execute("select count(*) from public.kb_chunks where doc_path like %s", (str(f.as_posix()),))
            c2 = cur.fetchone()[0]
            assert c2 == c1
