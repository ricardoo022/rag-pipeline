"""Shared pytest fixtures for the RAG test suite.

Mirrors ../ai-agent-sales/tests/conftest.py's pattern (db_ready / clean_db / require_*
fixtures), adapted to this repo: single sql/*.sql migration set (no CRM split), kb_docs/
kb_chunks truncation instead of CRM tables, and OpenRouter instead of OpenAI directly.
"""

import os
import pytest
import psycopg


@pytest.fixture(scope="session")
def db_ready():
    """Applies sql/*.sql once per test session and returns the DB URL.

    Skips (rather than fails) the dependent tests if DB_* isn't configured or the DB
    isn't reachable — keeps the suite runnable without a provisioned database.
    """
    try:
        from rag.db import get_db_url
        url = get_db_url()
    except RuntimeError as e:
        pytest.skip(str(e))

    try:
        from sql.migrate import migrate
        migrate()
    except Exception as e:
        pytest.skip(f"Could not apply migrations / connect to the DB: {e}")
    return url


@pytest.fixture(autouse=True)
def clean_db(db_ready):
    """Truncates kb_docs/kb_chunks before each test to isolate scenarios.

    Destructive (TRUNCATE) — fine for a dev/test database, not meant to run against
    anything shared/production.
    """
    from rag.ingest import truncate_kb_tables
    truncate_kb_tables()
    yield


@pytest.fixture(scope="session")
def require_embeddings():
    """Skips dependent tests unless OPENROUTER_API_KEY is configured (embeddings provider)."""
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not configured")
    return True


@pytest.fixture(scope="session")
def require_cohere():
    """Skips dependent tests unless COHERE_API_KEY is configured (reranking)."""
    if not os.getenv("COHERE_API_KEY"):
        pytest.skip("COHERE_API_KEY not configured")
    return True


@pytest.fixture(scope="session")
def llm_judge_yesno(require_embeddings):
    """Returns a callable(question, expected, contexts) -> bool LLM judge.

    Routed through OpenRouter like rag.ingest.get_embedder(), same provider choice.
    """
    from langchain_openai import ChatOpenAI

    def _judge(question: str, expected: str, contexts: list[str]) -> bool:
        joined = "\n\n---\n\n".join(contexts[:3])
        prompt = (
            "You are an evaluator. Given the CONTEXTS, answer only 'yes' or 'no': are they "
            "sufficient to answer the QUESTION and do they contain (directly or clearly) the "
            "EXPECTED concept/term?\n\n"
            f"QUESTION: {question}\n"
            f"EXPECTED: {expected}\n\n"
            f"CONTEXTS:\n{joined}\n\n"
            "Output: 'yes' or 'no' only."
        )
        llm = ChatOpenAI(
            model=os.getenv("EVAL_LLM_MODEL", "openai/gpt-4.1-mini"),
            temperature=0,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        out = (llm.invoke(prompt).content or "").strip().lower()
        return out.startswith("yes")

    return _judge
