"""Postgres connection shared by the whole RAG pipeline.

Provider-agnostic: works against any Postgres with the pgvector extension
enabled (Supabase, local Docker, self-hosted, RDS, ...). Reads connection
details from DB_* env vars — see .env.example.
"""

import os

import psycopg
from dotenv import load_dotenv

load_dotenv()


def get_db_url() -> str:
    user = os.getenv("DB_USER")
    pwd = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME")
    if not all([user, pwd, host, name]):
        raise RuntimeError(
            "Missing DB config: set DB_USER, DB_PASSWORD, DB_HOST, DB_NAME "
            "(optionally DB_PORT, DB_SSLMODE) in .env"
        )
    sslmode = os.getenv("DB_SSLMODE", "require")
    return f"postgresql://{user}:{pwd}@{host}:{port}/{name}?sslmode={sslmode}"


def get_conn():
    return psycopg.connect(get_db_url())
