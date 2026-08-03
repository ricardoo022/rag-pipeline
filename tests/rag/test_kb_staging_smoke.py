from pathlib import Path
import pytest

pytestmark = pytest.mark.phase0


def test_kb_staging_idempotent(tmp_path: Path, db_ready):
    from rag.ingest import stage_docs_from_dir
    import psycopg

    kb = tmp_path / "kb"
    kb.mkdir()
    f = kb / "doc.md"
    f.write_text("Line 1\n\nLine 2", encoding="utf-8")

    n1 = stage_docs_from_dir(str(kb), empresa="Test Company")
    assert n1 == 1
    n2 = stage_docs_from_dir(str(kb), empresa="Test Company")
    assert n2 == 1  # inserts attempted, but the second is a no-op via hash/unique

    with psycopg.connect(db_ready) as conn:
        with conn.cursor() as cur:
            cur.execute("select count(*) from public.kb_docs where source_path=%s", (str(f.as_posix()),))
            c = cur.fetchone()[0]
            assert c == 1, "Staging should keep only one version per identical (path, hash)"
