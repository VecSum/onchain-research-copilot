"""Command line entry point.

Three commands cover the whole workflow: list what a source can see, run one
research question end to end, and build the retrieval index. The research
command prints the artifact paths so a demo can be driven from a shell with
no surrounding tooling.
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import Config, ConfigError, load_config
from .models import Report
from .pipeline import list_tokens, run_research
from .render import render_html, render_json, render_markdown


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="config.toml", help="path to a TOML config")
    parser.add_argument("--offline", action="store_true",
                        help="force fixture data so the run needs no network")
    parser.add_argument("--chain", default="bsc", help="chain slug, default bsc")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="onchain-copilot",
                                     description="On-chain research copilot")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="list pools a source can see")
    _add_common(list_parser)
    list_parser.add_argument("--limit", type=int, default=10)

    research = sub.add_parser("research", help="run one research question")
    _add_common(research)
    research.add_argument("--pool", required=True, help="pool or pair address")
    research.add_argument("--question", required=True, help="the question to answer")
    research.add_argument("--out", default="output", help="artifact directory")
    research.add_argument("--print-report", action="store_true",
                            help="print the markdown to stdout after writing files")

    ingest = sub.add_parser("ingest", help="build the retrieval index and report its size")
    _add_common(ingest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.command == "list":
        mode = "fixtures" if args.offline else None
        try:
            rows = list_tokens(config, mode=mode, chain=args.chain, limit=args.limit)
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            print(f"no data available: {exc}", file=sys.stderr)
            return 1
        for row in rows:
            print(f"{row['symbol']:10} {row['pool_address'][:12]}  {row['dex']}")
        print(f"{len(rows)} pool(s) via {rows[0]['source'] if rows else 'no source'}")
        return 0

    if args.command == "ingest":
        from .retrieval import build_index, load_documents

        docs = load_documents(config.get("retrieval", "corpus_dir"))
        index = build_index(docs,
                             chunk_chars=int(config.get("retrieval", "chunk_chars")),
                             chunk_overlap=int(config.get("retrieval", "chunk_overlap")),
                             dim=int(config.get("retrieval", "dim")))
        print(f"{len(docs)} documents -> {len(index.chunks)} chunks")
        return 0 if index.chunks else 1

    if args.command == "research":
        mode = "fixtures" if args.offline else None
        try:
            outcome = run_research(
                args.pool, args.question, config=config, mode=mode,
                chain=args.chain, report_name=None,
            )
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            print(f"research failed: {exc}", file=sys.stderr)
            return 1
        artifacts = outcome.get("artifacts", {})
        for fmt, path in artifacts.items():
            print(f"{fmt}: {path}")
        if outcome.get("trace", {}).get("degraded"):
            print("degraded: " + ", ".join(outcome["trace"]["degraded"]))
        if args.print_report:
            print(outcome.get("_markdown", ""))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
