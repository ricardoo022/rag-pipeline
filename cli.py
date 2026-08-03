#!/usr/bin/env python3
"""Entrypoint for the RAG pipeline: migrate | ingest | search | eval."""

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

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
