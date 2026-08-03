from pathlib import Path
import pytest

pytestmark = pytest.mark.phase0


def test_stage_then_chunk(tmp_path: Path, db_ready, require_embeddings):
    from rag.ingest import stage_docs_from_dir, materialize_chunks_from_staging
    import psycopg

    kb = tmp_path / "kb"
    kb.mkdir()
    f = kb / "doc.md"
    f.write_text("""
# Product A — Company X
30/60 payment terms and 12-month warranty.
""".strip(), encoding="utf-8")

    empresa = "Company X"
    stage_docs_from_dir(str(kb), empresa=empresa)
    n_markdown = materialize_chunks_from_staging(strategy="markdown", empresa=empresa)
    assert n_markdown >= 1

    with psycopg.connect(db_ready) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "select count(*) from public.kb_chunks where lower(empresa)=lower(%s) and meta->>'chunking'='markdown'",
                (empresa,),
            )
            c_md = cur.fetchone()[0]
            assert c_md >= 1

    # Re-materialize under semantic too, straight from staging
    n_sem = materialize_chunks_from_staging(strategy="semantic", empresa=empresa)
    assert n_sem >= 1
    with psycopg.connect(db_ready) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "select count(*) from public.kb_chunks where lower(empresa)=lower(%s) and meta->>'chunking'='semantic'",
                (empresa,),
            )
            c_sem = cur.fetchone()[0]
            assert c_sem >= 1
