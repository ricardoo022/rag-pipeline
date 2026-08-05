#!/usr/bin/env python3
"""Phase 3 runner — agent E2E, judged by an LLM on the final answer.

Runs `agent/rag_agent.py`'s ReAct graph over `tests/rag/rag_queries.yaml` and writes
CSV/JSON results to `eval/analysis/`. Sibling to `eval/run.py` (which sweeps
retrieval-only experiments) — this one exercises the agent as a whole (tool calls +
final answer), not just retrieval.

Usage:
  PYTHONPATH=. .venv/bin/python eval/run_agent.py [--fast] [--max N] [--outfile NAME]

Examples:
  PYTHONPATH=. .venv/bin/python eval/run_agent.py --fast
  PYTHONPATH=. .venv/bin/python eval/run_agent.py --max 50 --outfile phase3_full

Outputs:
  eval/analysis/<NAME>_results.json   (summary + details)
  eval/analysis/<NAME>_results.csv    (summary + details in a single CSV)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import yaml

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def llm_judge_yesno_with_reason(question: str, expected: str, answer_text: str) -> tuple[bool, str]:
    """LLM as a binary (yes/no) judge with a short justification, over the agent's ANSWER
    (not retrieved contexts — that's what eval/run.py's judge looks at instead)."""
    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        return False, ""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return False, ""
    prompt = (
        "You are an evaluator. Analyze the ANSWER below and answer 'yes' or 'no' followed by a short justification.\n"
        "The output must be a SINGLE LINE in the format: <yes|no> — <short justification>.\n"
        "The justification should briefly cite the terms/snippets that support the decision and whether the QUESTION was answered.\n\n"
        f"QUESTION: {question}\n"
        f"EXPECTED: {expected}\n\n"
        f"ANSWER:\n{answer_text}\n\n"
        "Output: a single line in the format: yes — <justification> or no — <justification>."
    )
    try:
        llm = ChatOpenAI(
            model=os.getenv("EVAL_LLM_MODEL", "openai/gpt-4.1-mini"),
            temperature=0,
            openai_api_base=OPENROUTER_BASE_URL,
            openai_api_key=api_key,
        )
        raw = (llm.invoke(prompt).content or "").strip()
        low = raw.lower()
        ok = low.startswith("yes")
        just = raw
        for sep in ["—", "-", ":", "|", "–"]:
            if sep in raw:
                parts = raw.split(sep, 1)
                just = parts[1].strip() if len(parts) > 1 else raw
                break
        else:
            toks = raw.split(None, 1)
            just = toks[1].strip() if len(toks) > 1 else ""
        return ok, just
    except Exception:
        return False, ""


def extract_answer_text(last_content: Any) -> str:
    """Same content-block handling as cli.py's `chat` command / the phase3 pytest test."""
    if isinstance(last_content, list):
        parts: list[str] = []
        for it in last_content:
            if isinstance(it, dict) and it.get("type") in ("text", "output_text"):
                t = it.get("text") or ""
                if t:
                    parts.append(t)
        return "\n".join(parts)
    return last_content if isinstance(last_content, str) else str(last_content or "")


def load_queries() -> list[dict[str, Any]]:
    base = Path(__file__).resolve().parents[1] / "tests" / "rag"
    q_file = base / "rag_queries.yaml"
    assert q_file.exists(), f"Missing queries file: {q_file}"
    data = yaml.safe_load(q_file.read_text(encoding="utf-8")) or {}
    return data.get("queries") or []


def run(args: argparse.Namespace) -> int:
    from langchain_core.messages import HumanMessage

    from agent.rag_agent import graph

    queries = load_queries()
    if args.fast:
        queries = queries[:10]
    if args.max and args.max > 0:
        queries = queries[: args.max]

    # Informative only — the agent's kb_search tool fixes chunking=semantic internally.
    chunking = "semantic"

    out_dir = Path(__file__).resolve().parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    outname = args.outfile or "phase3"

    total = 0
    lyes = 0
    details: list[dict[str, Any]] = []
    for q in queries:
        pergunta = q.get("pergunta") or q.get("question")
        exp = q.get("answer") or q.get("expected") or ", ".join(q.get("expected_all") or [])
        if not (pergunta and exp):
            continue
        total += 1

        state = graph.invoke({"messages": [HumanMessage(content=pergunta)]})
        messages = state.get("messages") or []
        raw = messages[-1].content if messages and hasattr(messages[-1], "content") else ""
        answer = extract_answer_text(raw)

        try:
            ok, reason = llm_judge_yesno_with_reason(pergunta, exp, answer)
        except Exception:
            ok, reason = False, ""
        lyes += 1 if ok else 0

        details.append({
            "question": pergunta,
            "expected": exp,
            "answer": answer[:4000],
            "judge_yes": bool(ok),
            "judge_justification": reason,
        })

    rate = round(lyes / max(total, 1), 2)
    row = {
        "chunking": chunking,
        "total": total,
        "llm_yes": rate,
    }

    csv_path = out_dir / f"{outname}_results.csv"
    summary_fields = ["chunking", "total", "llm_yes"]
    detail_fields = ["question", "expected", "answer", "judge_yes", "judge_justification"]
    union_fields: list[str] = ["row_type"]
    for f in summary_fields + detail_fields:
        if f not in union_fields:
            union_fields.append(f)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=union_fields)
        w.writeheader()
        rsum = {k: row.get(k) for k in summary_fields}
        rsum["row_type"] = "summary"
        w.writerow(rsum)
        for d in details:
            dd = {k: d.get(k) for k in detail_fields}
            dd_row = {k: None for k in union_fields}
            dd_row.update(dd)
            dd_row["row_type"] = "detail"
            w.writerow(dd_row)

    json_path = out_dir / f"{outname}_results.json"
    json_path.write_text(json.dumps({"summary": [row], "details": details}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== PHASE 3 — AGENT (LLM-JUDGE ON DATASET) ===")
    print(f"chunking={chunking} | total={total} | llm_yes={rate:0.2f}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 3 runner (agent E2E, LLM-judge on final answer)")
    p.add_argument("--fast", action="store_true", help="Use only ~10 questions (shortcut; equivalent to RAG_FAST=1)")
    p.add_argument("--max", type=int, default=0, help="Limit to N questions (0 = all)")
    p.add_argument("--outfile", default="phase3", help="Output file prefix (default: phase3)")
    args = p.parse_args(argv)
    if args.fast:
        os.environ["RAG_FAST"] = "1"
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
