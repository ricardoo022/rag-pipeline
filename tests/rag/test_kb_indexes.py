import psycopg
import pytest

pytestmark = pytest.mark.phase0


def test_kb_indexes_exist(db_ready):
    """Confirms the KB indexes (HNSW, GIN, and support B-trees) exist."""
    with psycopg.connect(db_ready) as conn:
        with conn.cursor() as cur:
            names = {
                "ix_kb_chunks_embedding_hnsw",
                "ix_kb_chunks_fts_gin",
                "ix_kb_chunks_doc",
                "ix_kb_chunks_empresa",
                "ix_kb_chunks_client",
            }
            cur.execute(
                """
                select indexname
                from pg_indexes
                where schemaname = 'public' and tablename = 'kb_chunks'
                """
            )
            present = {r[0] for r in (cur.fetchall() or [])}
            missing = sorted(names - present)
            assert not missing, f"Missing indexes: {missing}"
