from pathlib import Path

import pytest
import yaml
from langchain_core.messages import HumanMessage

# Phase 3 — end-to-end agent evaluation, judged only on the final answer
pytestmark = pytest.mark.phase3


def _load_dataset() -> list[dict]:
    ds_path = Path(__file__).resolve().parents[0] / "rag_queries.yaml"
    assert ds_path.exists(), "Missing queries file 'tests/rag/rag_queries.yaml'"
    data = yaml.safe_load(ds_path.read_text(encoding="utf-8")) or {}
    return data.get("queries") or []


def _extract_answer(state: dict) -> str:
    """Same extraction cli.py's `chat` command uses (handles content-block lists)."""
    messages = state.get("messages") or []
    content = messages[-1].content if messages else ""
    if isinstance(content, list):
        content = "\n".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") in ("text", "output_text")
        )
    return content or ""


def test_phase3_agent_dataset_llm_judge(seeded_kb_empresa_xy, db_ready, require_embeddings, llm_judge_yesno):
    """Runs agent/rag_agent.py's ReAct agent over the shared dataset and judges only the
    final answer (LLM-judge), not the retrieved chunks.

    - Uses `seeded_kb_empresa_xy` (this repo's `clean_db` truncates kb_docs/kb_chunks before
      every test, unlike ../ai-agent-sales', so the KB has to be (re)materialized per test
      rather than assumed pre-populated by `make rag_phase0`).
    - The agent's kb_search tool fixes chunking=semantic and doesn't filter by empresa/client_id
      (see agent/rag_agent.py), so it searches across both seeded empresas.
    - RAG_FAST=1 limits to ~10 questions; RAG_MAX_Q=<n> limits to N.
    """
    import os

    from agent.rag_agent import graph

    queries = _load_dataset()

    max_q = None
    if os.getenv("RAG_FAST", "").strip().lower() in ("1", "true", "yes", "on"):
        max_q = 10
    elif os.getenv("RAG_MAX_Q", "").strip().isdigit():
        max_q = int(os.getenv("RAG_MAX_Q"))
    if max_q:
        queries = queries[:max_q]

    total = 0
    lyes = 0
    rows = []
    for q in queries:
        pergunta = q.get("pergunta") or q.get("question")
        exp_all = q.get("expected_all")
        exp_one = q.get("expected")
        judge_expected = q.get("answer") or exp_one or (", ".join(exp_all) if exp_all else None)
        if not (pergunta and judge_expected):
            continue
        total += 1

        state = graph.invoke({"messages": [HumanMessage(content=pergunta)]})
        answer = _extract_answer(state)

        try:
            ok = llm_judge_yesno(pergunta, judge_expected, [answer])
        except Exception:
            ok = False
        lyes += 1 if ok else 0
        rows.append((pergunta, judge_expected, answer, ok))

    rate = round(lyes / max(total, 1), 2)
    print("\n=== PHASE 3 — AGENT (LLM-JUDGE ON DATASET) ===")
    print(f"chunking=semantic | total={total} | llm_yes={rate:0.2f}")
    print("\nix | pergunta                               | esperado             | resposta do agente (snippet)          | ok")
    print("-" * 130)
    for i, (pergunta, expected, answer, ok) in enumerate(rows, start=1):
        qshort = (pergunta[:35] + "…") if len(pergunta) > 35 else pergunta
        exps = (str(expected)[:18] + "…") if len(str(expected)) > 18 else str(expected)
        asnip = (answer[:38] + "…") if len(answer) > 39 else answer
        print(f"{i:>2d} | {qshort:35s} | {exps:20s} | {asnip:39s} | {'OK' if ok else ' X'}")

    assert rate >= 0.5, f"Insufficient agent performance (llm_yes={rate:0.2f} < 0.5)"
