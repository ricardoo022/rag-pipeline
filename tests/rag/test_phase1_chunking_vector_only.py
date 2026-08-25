from pathlib import Path
import uuid
import pytest

# Phase 1 — chunking comparison using vector search only
pytestmark = pytest.mark.phase0


def test_phase1_chunking_vector_only(tmp_path: Path, db_ready, require_embeddings):
    """Compares fixed vs markdown vs semantic (VECTOR only).

    Phase 1 goal: quickly see how chunking impacts vector retrieval (no hybrids,
    no LLM judge).
    """
    from rag.ingest import stage_docs_from_dir, materialize_chunks_from_staging
    from rag.search import kb_search_client

    # Builds a mini-KB for Empresa X and Empresa Y (noise)
    kb = tmp_path / "kb"
    # Creates three copies of the content (one per strategy) to avoid (doc_path, chunk_ix) conflicts
    (kb / "x_fixed").mkdir(parents=True)
    (kb / "x_markdown").mkdir(parents=True)
    (kb / "x_semantic").mkdir(parents=True)
    (kb / "y_fixed").mkdir(parents=True)
    (kb / "y_markdown").mkdir(parents=True)
    (kb / "y_semantic").mkdir(parents=True)

    def _w(p, txt):
        p.write_text(txt.strip(), encoding="utf-8")

    _w(kb / "x_fixed" / "produto_a.md",
        """
# Product A — Policies (Empresa X)
Standard 12-month warranty. Installment payment over 30/60 days.
"""
    )
    _w(kb / "x_markdown" / "produto_a.md",
        """
# Product A — Policies (Empresa X)
Standard 12-month warranty. Installment payment over 30/60 days.
"""
    )
    _w(kb / "x_semantic" / "produto_a.md",
        """
# Product A — Policies (Empresa X)
Standard 12-month warranty. Installment payment over 30/60 days.
"""
    )

    _w(kb / "x_fixed" / "servico_b.md",
        """
# Service B — SLA (Empresa X)
The target SLA is 99.9% monthly uptime.
"""
    )
    _w(kb / "x_markdown" / "servico_b.md",
        """
# Service B — SLA (Empresa X)
The target SLA is 99.9% monthly uptime.
"""
    )
    _w(kb / "x_semantic" / "servico_b.md",
        """
# Service B — SLA (Empresa X)
The target SLA is 99.9% monthly uptime.
"""
    )

    _w(kb / "x_fixed" / "cancelamento.md",
        """
# Cancellation Policy (Empresa X)
Cancellation allowed within 7 business days after signing.
"""
    )
    _w(kb / "x_markdown" / "cancelamento.md",
        """
# Cancellation Policy (Empresa X)
Cancellation allowed within 7 business days after signing.
"""
    )
    _w(kb / "x_semantic" / "cancelamento.md",
        """
# Cancellation Policy (Empresa X)
Cancellation allowed within 7 business days after signing.
"""
    )

    _w(kb / "x_fixed" / "financeiro.md",
        """
# Financial Terms (Empresa X)
5% cash discount. Installments over 30/60/90 days for orders above R$ 10,000.
"""
    )
    _w(kb / "x_markdown" / "financeiro.md",
        """
# Financial Terms (Empresa X)
5% cash discount. Installments over 30/60/90 days for orders above R$ 10,000.
"""
    )
    _w(kb / "x_semantic" / "financeiro.md",
        """
# Financial Terms (Empresa X)
5% cash discount. Installments over 30/60/90 days for orders above R$ 10,000.
"""
    )

    _w(kb / "x_fixed" / "suporte.md",
        """
# Support (Empresa X)
24/7 support via chat and email. Target first-response time: 2 hours.
"""
    )
    _w(kb / "x_markdown" / "suporte.md",
        """
# Support (Empresa X)
24/7 support via chat and email. Target first-response time: 2 hours.
"""
    )
    _w(kb / "x_semantic" / "suporte.md",
        """
# Support (Empresa X)
24/7 support via chat and email. Target first-response time: 2 hours.
"""
    )

    _w(kb / "x_fixed" / "implantacao.md",
        """
# Deployment (Empresa X)
Typical implementation timeline: 5 business days. We offer remote training.
"""
    )
    _w(kb / "x_markdown" / "implantacao.md",
        """
# Deployment (Empresa X)
Typical implementation timeline: 5 business days. We offer remote training.
"""
    )
    _w(kb / "x_semantic" / "implantacao.md",
        """
# Deployment (Empresa X)
Typical implementation timeline: 5 business days. We offer remote training.
"""
    )

    _w(kb / "x_fixed" / "troca.md",
        """
# Exchange Policy (Empresa X)
Exchanges allowed within 30 calendar days with a receipt.
"""
    )
    _w(kb / "x_markdown" / "troca.md",
        """
# Exchange Policy (Empresa X)
Exchanges allowed within 30 calendar days with a receipt.
"""
    )
    _w(kb / "x_semantic" / "troca.md",
        """
# Exchange Policy (Empresa X)
Exchanges allowed within 30 calendar days with a receipt.
"""
    )

    _w(kb / "x_fixed" / "seguranca.md",
        """
# Security and Compliance (Empresa X)
AES-256 encryption at rest. Daily backups retained for 30 days.
"""
    )
    _w(kb / "x_markdown" / "seguranca.md",
        """
# Security and Compliance (Empresa X)
AES-256 encryption at rest. Daily backups retained for 30 days.
"""
    )
    _w(kb / "x_semantic" / "seguranca.md",
        """
# Security and Compliance (Empresa X)
AES-256 encryption at rest. Daily backups retained for 30 days.
"""
    )
    # Empresa Y (noise)
    _w(kb / "y_fixed" / "produto_a.md",
        """
# Product A — Policies (Empresa Y)
No mention of a 12-month warranty or 30/60 installments.
"""
    )
    _w(kb / "y_markdown" / "produto_a.md",
        """
# Product A — Policies (Empresa Y)
No mention of a 12-month warranty or 30/60 installments.
"""
    )
    _w(kb / "y_semantic" / "produto_a.md",
        """
# Product A — Policies (Empresa Y)
No mention of a 12-month warranty or 30/60 installments.
"""
    )
    _w(kb / "y_fixed" / "financeiro.md",
        """
# Financial Terms (Empresa Y)
No mention of a cash discount or 30/60/90 installments.
"""
    )
    _w(kb / "y_markdown" / "financeiro.md",
        """
# Financial Terms (Empresa Y)
No mention of a cash discount or 30/60/90 installments.
"""
    )
    _w(kb / "y_semantic" / "financeiro.md",
        """
# Financial Terms (Empresa Y)
No mention of a cash discount or 30/60/90 installments.
"""
    )
    _w(kb / "y_fixed" / "suporte.md",
        """
# Support (Empresa Y)
Support during business hours.
"""
    )
    _w(kb / "y_markdown" / "suporte.md",
        """
# Support (Empresa Y)
Support during business hours.
"""
    )
    _w(kb / "y_semantic" / "suporte.md",
        """
# Support (Empresa Y)
Support during business hours.
"""
    )

    # Isolates with a unique empresa to avoid interference from previous runs
    empresa = f"Empresa X {str(uuid.uuid4())[:8]}"
    empresa_y = f"Empresa Y {str(uuid.uuid4())[:8]}"

    # 1) Staging: simulates the real world — a single base per empresa
    #    (uses only the markdown-dir content as the base; all three x_* dirs have the same content)
    stage_docs_from_dir(str(kb / "x_markdown"), empresa=empresa)
    stage_docs_from_dir(str(kb / "y_markdown"), empresa=empresa_y)

    # 2) Materializes chunks per strategy from the SAME staging (base empresa), differentiating
    #    by meta.chunking and doc_path_prefix so writes don't collide
    materialize_chunks_from_staging(strategy="fixed", empresa=empresa, doc_path_prefix="fixed")
    materialize_chunks_from_staging(strategy="markdown", empresa=empresa, doc_path_prefix="markdown")
    materialize_chunks_from_staging(strategy="semantic", empresa=empresa, doc_path_prefix="semantic")

    # Queries and expected keywords
    cases = [
        ("What is the warranty for Product A?", "12-month"),
        ("What are the payment terms for Product A?", "30/60"),
        ("What is the maximum number of installments offered?", "90"),
        ("Is there a cash discount?", "5%"),
        ("Is support available 24/7?", "24/7"),
        ("What is the first-response time?", "2 hours"),
        ("What is the implementation timeline?", "5 business days"),
        ("Is remote training available?", "remote training"),
        ("What is the cancellation policy?", "7 business days"),
        ("What is the exchange policy?", "30 calendar days"),
        ("How is encryption at rest handled?", "AES-256"),
        ("How long are backups kept?", "30 days"),
        ("What is the SLA for Service B?", "99.9%"),
    ]

    strategies = (
        ("fixed", empresa),
        ("markdown", empresa),
        ("semantic", empresa),
    )
    results = []
    # Per-question feedback (hit@1) for each chunking strategy
    feedback = []
    total = len(cases)
    for chunking, empresa_lookup in strategies:
        hits5 = 0
        hits1 = 0
        per_q = []  # (ok1, doc#chunk, content_snippet)
        for q, expected in cases:
            res = kb_search_client.invoke({
                "query": q,
                "k": 5,
                "search_type": "vector",
                "reranker": "none",
                "empresa": empresa_lookup,
                "chunking": chunking,
            })
            norm = expected.lower()
            got5 = any(norm in (it["content"] or "").lower() for it in res)
            got1 = False
            docloc = "-"
            snippet = ""
            if res:
                top = res[0]
                c = (top["content"] or "")
                got1 = norm in c.lower()
                from pathlib import Path as _P
                docloc = f"{_P(top['doc_path']).name}:{top['chunk_ix']}"
                snippet = c[:80].replace("\n", " ")
            hits5 += 1 if got5 else 0
            hits1 += 1 if got1 else 0
            per_q.append((got1, docloc, snippet))
        results.append((chunking, hits5 / total, hits1 / total))
        feedback.append((chunking, per_q))

    print("\n=== PHASE 1 — CHUNKING COMPARISON (VECTOR ONLY) ===")
    print("chunking  | hit@5 | hit@1")
    print("-" * 34)
    for chunking, hit5, hit1 in results:
        print(f"{chunking:8s}| {hit5:5.2f} | {hit1:5.2f}")

    # At least one strategy should perform reasonably well
    assert any(hit5 >= 0.5 for _, hit5, _ in results), "No strategy reached 50% hit@5 in VECTOR"

    # Per-question feedback (hit@1) per chunking
    print("\n=== PHASE 1 — PER-QUESTION FEEDBACK (hit@1 per chunking) ===")
    print("ix | question                                | expected        | fixed        | markdown     | semantic     ")
    print("-" * 106)
    # Reorganizes feedback by question (rows) and chunking (columns)
    # feedback = [(chunking, [(ok1, docloc, snippet), ...])]
    per_question_rows = []
    for i, (q, expected) in enumerate(cases, start=1):
        row = {"ix": i, "q": q, "expected": expected, "fixed": (False, "-", ""), "markdown": (False, "-", ""), "semantic": (False, "-", "")}
        for chunking, per_q in feedback:
            ok1, docloc, snippet = per_q[i - 1]
            row[chunking] = (ok1, docloc, snippet)
        per_question_rows.append(row)

    def _cell(val):
        ok, docloc, _snip = val
        return ("OK" if ok else " X") + f" {docloc:12s}"

    for row in per_question_rows:
        qshort = (row["q"][:35] + "…") if len(row["q"]) > 35 else row["q"]
        exps = (row["expected"][:14] + "…") if len(row["expected"]) > 14 else row["expected"]
        print(f"{row['ix']:>2d} | {qshort:35s} | {exps:14s} | {_cell(row['fixed'])} | {_cell(row['markdown'])} | {_cell(row['semantic'])}")

    # Optional debug: prints per-empresa/chunking counts to confirm materialization
    try:
        import psycopg
        with psycopg.connect(db_ready) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "select coalesce(empresa,'<null>') as empresa, meta->>'chunking' as chunking, count(*)\n"
                    "from public.kb_chunks where lower(empresa)=lower(%s) group by 1,2 order by 1,2",
                    (empresa,),
                )
                rows = cur.fetchall() or []
                print("\n[DEBUG] kb_chunks per empresa/chunking:")
                for emp, strat, c in rows:
                    print(f"  {emp} | {strat} -> {c}")
    except Exception:
        pass
