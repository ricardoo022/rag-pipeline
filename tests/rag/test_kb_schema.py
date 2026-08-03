import pytest

pytestmark = pytest.mark.phase0


def test_kb_schema_migration(db_ready):
    """Applies sql/*.sql and confirms the run reports every file applied."""
    from sql.migrate import discover_sql_files, apply_sql_files

    files = discover_sql_files()
    assert files, "No .sql files found in sql/"
    applied = apply_sql_files(files)
    assert applied == len(files)
