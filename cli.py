#!/usr/bin/env python3
"""Entrypoint for the RAG pipeline: migrate | ingest | search | eval | chat."""

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("migrate", help="Apply sql/*.sql against the configured database")

    sub.add_parser("truncate", help="Truncate kb_docs/kb_chunks (destructive) so you can re-ingest from scratch")

    p_ingest = sub.add_parser("ingest", help="Stage kb/*.md into kb_docs, then chunk+embed into kb_chunks")
    p_ingest.add_argument("--base-dir", default="kb", help="Directory of .md files to ingest (default: kb)")
    p_ingest.add_argument("--strategy", default="fixed", choices=["fixed", "markdown", "semantic"], help="Chunking strategy (default: fixed)")
    p_ingest.add_argument("--empresa", default=None, help="Optional empresa tag for staged docs/chunks")
    p_ingest.add_argument("--client-id", default=None, help="Optional client_id (uuid) tag")
    p_ingest.add_argument("--chunk-size", type=int, default=800)
    p_ingest.add_argument("--chunk-overlap", type=int, default=200)
    p_ingest.add_argument("--skip-stage", action="store_true", help="Skip staging, only (re)materialize chunks")
    p_ingest.add_argument("--skip-chunks", action="store_true", help="Skip chunking, only stage docs")
    p_ingest.add_argument("--truncate", action="store_true", help="Truncate kb_docs/kb_chunks before running (destructive)")

    p_search = sub.add_parser("search", help="Query kb_chunks via rag/search.py's kb_search_client")
    p_search.add_argument("query", help="Search query text")
    p_search.add_argument("--k", type=int, default=5)
    p_search.add_argument("--search-type", default="hybrid", choices=["vector", "text", "hybrid", "hybrid_rrf", "hybrid_union"])
    p_search.add_argument("--reranker", default="none", choices=["none", "cohere"])
    p_search.add_argument("--rerank-candidates", type=int, default=24)
    p_search.add_argument("--client-id", default=None)
    p_search.add_argument("--empresa", default=None)
    p_search.add_argument("--chunking", default=None, choices=["fixed", "markdown", "semantic"])
    p_search.add_argument("--use-hyde", action="store_true")
    p_search.add_argument("--match-threshold", type=float, default=None)

    p_eval = sub.add_parser("eval", help="Run eval/run.py's experiments sweep against the materialized KB")
    p_eval.add_argument("--fast", action="store_true", help="Use only ~10 questions")
    p_eval.add_argument("--max", type=int, default=0, help="Limit to N questions (0 = all)")
    p_eval.add_argument("--outfile", default="phase2", help="Output file prefix (default: phase2)")

    p_chat = sub.add_parser("chat", help="Ask the ReAct RAG agent (agent/rag_agent.py) a single question")
    p_chat.add_argument("question", help="Question for the agent")

    args = parser.parse_args(argv)

    if args.command == "migrate":
        from sql.migrate import migrate
        return migrate()

    if args.command == "truncate":
        from rag.ingest import truncate_kb_tables
        truncate_kb_tables()
        print("Truncated kb_docs and kb_chunks.")
        return 0

    if args.command == "ingest":
        from rag.ingest import graph, truncate_kb_tables

        if args.truncate:
            truncate_kb_tables()

        result = graph.invoke({
            "base_dir": args.base_dir,
            "strategy": args.strategy,
            "empresa": args.empresa,
            "client_id": args.client_id,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "skip_stage": args.skip_stage,
            "skip_chunks": args.skip_chunks,
        })
        print(f"Staged {result.get('staged', 0)} doc(s), materialized {result.get('chunked', 0)} chunk(s).")
        return 0

    if args.command == "search":
        import json

        from rag.search import kb_search_client

        results = kb_search_client.invoke({
            "query": args.query,
            "k": args.k,
            "search_type": args.search_type,
            "reranker": args.reranker,
            "rerank_candidates": args.rerank_candidates,
            "client_id": args.client_id,
            "empresa": args.empresa,
            "chunking": args.chunking,
            "use_hyde": args.use_hyde,
            "match_threshold": args.match_threshold,
        })
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    if args.command == "eval":
        from eval.run import main as eval_main

        return eval_main([
            *(["--fast"] if args.fast else []),
            "--max", str(args.max),
            "--outfile", args.outfile,
        ])

    if args.command == "chat":
        from langchain_core.messages import HumanMessage

        from agent.rag_agent import graph

        state = graph.invoke({"messages": [HumanMessage(content=args.question)]})
        messages = state.get("messages") or []
        content = messages[-1].content if messages else ""
        if isinstance(content, list):
            content = "\n".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") in ("text", "output_text")
            )
        print(content)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
