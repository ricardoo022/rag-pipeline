"""Apply .sql files from sql/ against the configured database, in filename order."""

from pathlib import Path

from rag.db import get_db_url
import psycopg

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"


def discover_sql_files() -> list[Path]:
    return sorted(SQL_DIR.glob("*.sql"))


def apply_sql_files(files: list[Path]) -> int:
    applied = 0
    with psycopg.connect(get_db_url()) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            for path in files:
                print(f"=== Applying: {path.name} ===")
                cur.execute(path.read_text(encoding="utf-8"))
                applied += 1
                print(f"OK: {path.name}")
    return applied


def migrate() -> int:
    files = discover_sql_files()
    if not files:
        print(f"No .sql files found in {SQL_DIR}")
        return 0
    applied = apply_sql_files(files)
    print(f"\nDone. Applied {applied}/{len(files)} file(s).")
    return 0
